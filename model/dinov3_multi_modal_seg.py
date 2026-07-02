"""
DINOv3 based multi-modal medical segmentation modules.

This file provides the following building blocks that are shared across the
project:

1. ``DINOv3Encoder`` – wraps a ViT/DINOv3 style backbone, optionally
   loads pretrained weights, freezes the encoder, and exposes intermediate
   token maps for Feature Pyramid style decoding.
2. ``MultiModalFusion`` – learns attention weights across an arbitrary number
   of modalities while keeping channel dimensions aligned.
3. ``FeaturePyramidDecoder`` – gradually upsamples fused tokens back to
   high-resolution 2D feature maps.
4. ``CombinedLoss`` – Dice + Cross-Entropy hybrid loss with optional class
   re-weighting.
5. ``DINOv3MultiModalSeg`` – end-to-end segmentation model that ties
   the above components together.

The implementation follows the architecture documented in
``md/MODEL_DESIGN_DOCUMENTATION.md`` and is compatible with the interfaces
used throughout ``train.py`` as well as ``model/multi_task_segmentation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import DINOv3 from embedded source (model/dinov3_src/)
from .dinov3_src.models import (
    DinoVisionTransformer,
    vit_base,
    vit_large,
    vit_huge2,
    vit_giant2,
)

def _build_backbone(model_name: str, image_size: int) -> DinoVisionTransformer:
    """Factory helper that instantiates the official DINOv3 ViT."""
    name = model_name.lower()
    # Critical Fix: Enable LayerScale, Storage Tokens, and Masked Bias to match pretrained DINOv3 weights.
    # Without these, 'ls1.gamma' and 'ls2.gamma' are ignored (effectively 1.0), causing activation explosion.
    kwargs = dict(
        img_size=image_size, 
        patch_size=16, 
        block_chunks=0,
        layerscale_init=1e-5,  # Enable LayerScale
        n_storage_tokens=4,    # Match standard DINOv3 registers
        mask_k_bias=True       # Match bias_mask in checkpoint
    )
    
    if "vitb16" in name:
        return vit_base(**kwargs)
    elif "vitl16" in name:
        return vit_large(**kwargs)
    elif "vith14" in name:
        kwargs['patch_size'] = 14
        return vit_huge2(**kwargs)
    elif "vitg14" in name:
        kwargs['patch_size'] = 14
        return vit_giant2(**kwargs)
    else:
        print(f"[DINOv3Encoder] Unknown model name {model_name}, defaulting to vit_base")
        return vit_base(**kwargs)


def _load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Loads DINOv3 checkpoint directly without conversion."""
    if not checkpoint_path:
        return
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"[DINOv3Encoder] Warning: Pretrained weights not found at {checkpoint_path}")
        return

    print(f"[DINOv3Encoder] Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(str(ckpt_file), map_location="cpu")
    
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove module/backbone prefixes if present
        k = k.replace("module.", "").replace("backbone.", "")
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    # Filter harmless missing keys
    missing_filtered = [k for k in missing if not k.startswith('head.')]
    unexpected_filtered = [k for k in unexpected if not k.startswith('head.')]
    
    if missing_filtered:
        print(f"[DINOv3Encoder] Missing keys ({len(missing_filtered)}): {missing_filtered[:5]}...")
    else:
        print(f"[DINOv3Encoder] ✓ All encoder weights loaded successfully!")
    
    if unexpected_filtered:
        print(f"[DINOv3Encoder] Unexpected keys ({len(unexpected_filtered)}): {unexpected_filtered[:5]}...")


class DINOv3Encoder(nn.Module):
    """Wrapper around a ViT/DINOv3 backbone that exposes patch features.

    Args:
        model_name: Backbone identifier (dinov3_vitb16, dinov3_vitl16, dinov3_vith14).
        pretrained_weights: Optional checkpoint path for DINOv3 weights.
        freeze: Whether to freeze the encoder parameters.
        unfreeze_last_n_blocks: Number of final transformer blocks to keep trainable.
        extract_layers: Transformer block indices whose outputs will be stored for
            decoder consumption (1-indexed, e.g., [8, 16, 20, 24]).
        image_size: Input spatial resolution (default: 224).
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained_weights: Optional[str] = None,
        freeze: bool = True,
        unfreeze_last_n_blocks: int = 0,
        extract_layers: Sequence[int] = (8, 16, 20, 24),
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.backbone: DinoVisionTransformer = _build_backbone(model_name, image_size)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.spatial_size = image_size // self.patch_size
        self.num_patches = self.spatial_size * self.spatial_size

        # Load checkpoint
        _load_checkpoint(self.backbone, pretrained_weights)
        
        max_layers = len(self.backbone.blocks)
        sanitized_layers = [
            int(layer)
            for layer in extract_layers
            if 1 <= int(layer) <= max_layers
        ]
        if not sanitized_layers:
            sanitized_layers = [max_layers]
        self.extract_layers = sorted(set(sanitized_layers))

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optionally unfreeze the last N transformer blocks for fine-tuning.
        if unfreeze_last_n_blocks > 0:
            # Unfreeze blocks
            for block in self.backbone.blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Unfreeze final norm
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            # Unfreeze cls_norm if exists
            if self.backbone.cls_norm is not None:
                for param in self.backbone.cls_norm.parameters():
                    param.requires_grad = True

    def forward(
        self,
        x: Tensor,
        return_intermediate: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Forward pass returning the final feature map and optional intermediates."""
        
        # Prepare indices for get_intermediate_layers
        # We want the layers specified in extract_layers (1-based)
        # Plus the last layer if not included
        
        target_indices = [i - 1 for i in self.extract_layers]
        last_layer_idx = len(self.backbone.blocks) - 1
        
        # We need a unique sorted list of indices to query
        query_indices = sorted(list(set(target_indices + [last_layer_idx])))
        
        # get_intermediate_layers with reshape=True returns (B, C, H, W) tensors
        # Note: DINOv3 expects indices relative to the end if n is int, but absolute if n is list?
        # Let's check source code logic:
        # "blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n"
        # So if we pass a list, it treats them as absolute indices (0-based). Correct.
        
        features = self.backbone.get_intermediate_layers(
            x, 
            n=query_indices, 
            reshape=True,
            return_class_token=False,
            norm=True
        )
        
        # Map returned features back to their indices
        idx_to_feat = {idx: feat for idx, feat in zip(query_indices, features)}
        
        final_map = idx_to_feat[last_layer_idx]
        
        if not return_intermediate:
            return final_map

        pyramid: Dict[str, Tensor] = {}
        for user_layer_idx in self.extract_layers:
            internal_idx = user_layer_idx - 1
            if internal_idx in idx_to_feat:
                pyramid[f"layer_{user_layer_idx}"] = idx_to_feat[internal_idx]
                
        return final_map, pyramid


class MultiModalFusion(nn.Module):
    """Modal-aware attention that fuses feature maps across modalities."""

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 1,
        attention_hidden_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = max(1, num_modalities)

        self.lateral_projs = nn.ModuleList(
            [
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                for _ in range(self.num_modalities)
            ]
        )

        attn_hidden = max(1, embed_dim // attention_hidden_ratio)
        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: List[Union[Tensor, Dict[str, Tensor]]]
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if not features:
            raise ValueError("MultiModalFusion requires at least one feature tensor.")

        # Handle dict inputs (multi-scale pyramid) recursively.
        if isinstance(features[0], dict):
            all_keys = set().union(*(feat.keys() for feat in features))
            fused: Dict[str, Tensor] = {}
            for key in sorted(all_keys):
                per_level = [feat[key] for feat in features if key in feat]
                fused[key] = self._fuse_single_level(per_level)
            return fused

        return self._fuse_single_level(features)  # type: ignore[arg-type]

    def _fuse_single_level(self, tensors: List[Tensor]) -> Tensor:
        if not tensors:
            raise ValueError("Received empty tensor list for fusion.")
        if len(tensors) == 1:
            return tensors[0]

        projected = []
        for idx, feat in enumerate(tensors):
            proj = self.lateral_projs[min(idx, self.num_modalities - 1)]
            projected.append(proj(feat))

        stacked = torch.stack(projected, dim=1)  # (B, M, C, H, W)
        pooled = stacked.mean(dim=(-1, -2))  # (B, M, C)

        b, m, c = pooled.shape
        attn_logits = self.attention_mlp(pooled.reshape(-1, c)).reshape(b, m, 1)
        weights = torch.softmax(attn_logits, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, M, 1, 1, 1)

        fused = (stacked * weights).sum(dim=1)
        return self.dropout(fused)


class UpsampleBlock(nn.Module):
    """ConvTranspose based upsample block with Dropout for regularization."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),  # 添加Dropout防止过拟合
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),  # 添加Dropout防止过拟合
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeaturePyramidDecoder(nn.Module):
    """Aggregates multi-layer encoder features and upsamples to high resolution."""

    def __init__(
        self,
        embed_dim: int,
        pyramid_layers: Sequence[str],
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        dropout: float = 0.3,  # 添加dropout参数
    ) -> None:
        super().__init__()
        if not decoder_channels:
            raise ValueError("decoder_channels must contain at least one entry.")

        self.pyramid_layers = list(pyramid_layers)
        self.base_channels = decoder_channels[0]
        self.lateral_convs = nn.ModuleDict(
            {
                layer: nn.Sequential(
                    nn.Conv2d(embed_dim, self.base_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.base_channels),
                    nn.GELU(),
                )
                for layer in self.pyramid_layers
            }
        )

        upsample_blocks: List[nn.Module] = []
        in_ch = self.base_channels
        for out_ch in decoder_channels:
            upsample_blocks.append(UpsampleBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.upsample = nn.Sequential(*upsample_blocks)
        self.out_channels = decoder_channels[-1]

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        if "layer_last" not in features:
            raise ValueError("Feature dictionary must contain 'layer_last'.")

        fused: Optional[Tensor] = None
        valid = 0
        for layer in self.pyramid_layers:
            if layer not in features:
                continue
            lateral = self.lateral_convs[layer](features[layer])
            fused = lateral if fused is None else fused + lateral
            valid += 1

        if fused is None:
            fused = self.lateral_convs["layer_last"](features["layer_last"])
            valid = 1

        fused = fused / max(1, valid)
        return self.upsample(fused)


class CombinedLoss(nn.Module):
    """Hybrid Dice + Cross-Entropy/Focal loss with options.

    Options:
    - foreground_only_dice: compute Dice only on foreground class (1) for binary tasks.
    - use_focal: replace CE with Focal loss (supports class_weights as alpha).
    - focal_gamma: focusing parameter for Focal loss.
    - class_weights: per-class weights (also acts as alpha for focal).
    """

    def __init__(
        self,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        class_weights: Optional[Tensor] = None,
        foreground_only_dice: bool = False,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is None else class_weights.clone().detach(),
            persistent=False,
        )
        self.eps = epsilon
        self.foreground_only_dice = foreground_only_dice
        self.use_focal = use_focal
        self.focal_gamma = float(focal_gamma)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        dice = self._dice_loss(logits, targets)
        if self.use_focal:
            ce = self._focal_ce_loss(logits, targets)
        else:
            ce = F.cross_entropy(
                logits,
                targets.long(),
                weight=self.class_weights,
            )
        return self.dice_weight * dice + self.ce_weight * ce

    def _dice_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 🔧 P0修复：改为只在空间维度求和，保留batch维，避免样本间混合
        dims = (2, 3)  # 改为只在H, W维度求和
        if self.foreground_only_dice and num_classes >= 2:
            probs_fg = probs[:, 1:2, ...]  # [B, 1, H, W]
            target_fg = target_one_hot[:, 1:2, ...]  # [B, 1, H, W]
            inter = torch.sum(probs_fg * target_fg, dims)  # [B, 1]
            denom = torch.sum(probs_fg + target_fg, dims)  # [B, 1]
            dice = (2.0 * inter + self.eps) / (denom + self.eps)  # [B, 1]
            return 1.0 - dice.mean()  # 在batch维取均值
        else:
            intersection = torch.sum(probs * target_one_hot, dims)  # [B, C]
            denominator = torch.sum(probs + target_one_hot, dims)  # [B, C]
            dice = (2.0 * intersection + self.eps) / (denominator + self.eps)  # [B, C]
            return 1.0 - dice.mean()  # 先在类别维再在batch维取均值

    def _focal_ce_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Compute per-pixel CE
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        # Gather p_t and log_p_t for true class
        targets_long = targets.long()
        p_t = probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
        log_p_t = log_probs.gather(1, targets_long.unsqueeze(1)).squeeze(1)
        # Alpha weighting per class if provided
        if self.class_weights is not None:
            alpha_t = self.class_weights[targets_long]
        else:
            alpha_t = torch.ones_like(p_t)
        loss = -alpha_t * ((1.0 - p_t) ** self.focal_gamma) * log_p_t
        return loss.mean()


class DINOv3MultiModalSeg(nn.Module):
    """End-to-end segmentation model built on top of a DINOv3 encoder."""

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained_weights: Optional[str] = None,
        num_classes: int = 2,
        num_modalities: int = 1,
        in_channels: int = 3,  # 新增：支持自定义输入通道数 (例如 2.5D 的 5层输入)
        freeze_encoder: bool = True,
        unfreeze_last_n_blocks: int = 0,
        extract_layers: Sequence[int] = (8, 16, 20, 24),
        decoder_channels: Sequence[int] = (512, 256, 128, 64),
        fusion_dropout: float = 0.1,
        decoder_dropout: float = 0.3,
        image_size: int = 224,
        use_image_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_modalities = max(1, num_modalities)
        self.use_image_adapter = bool(use_image_adapter)

        self.encoder = DINOv3Encoder(
            model_name=model_name,
            pretrained_weights=pretrained_weights,
            freeze=freeze_encoder,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            extract_layers=extract_layers,
            image_size=image_size,
        )

        # Ensure the decoder always sees the last layer in addition to the extracted set.
        pyramid_layers = list(dict.fromkeys(f"layer_{idx}" for idx in extract_layers))
        if "layer_last" not in pyramid_layers:
            pyramid_layers.append("layer_last")

        self.fusion = MultiModalFusion(
            embed_dim=self.encoder.embed_dim,
            num_modalities=self.num_modalities,
            dropout=fusion_dropout,
        )
        self.decoder = FeaturePyramidDecoder(
            embed_dim=self.encoder.embed_dim,
            pyramid_layers=pyramid_layers,
            decoder_channels=decoder_channels,
            dropout=decoder_dropout,
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, self.decoder.out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.decoder.out_channels // 2),
            nn.GELU(),
            nn.Conv2d(self.decoder.out_channels // 2, num_classes, kernel_size=1),
        )

        # Image adapter to map custom input channels (e.g. 5 slices) to ViT-friendly space (3 channels)
        if self.use_image_adapter or in_channels != 3:
            # 强制启用adapter如果输入通道不为3
            self.image_adapter = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 3, kernel_size=1, bias=False),
            )
        else:
            self.image_adapter = None

    def forward(
        self,
        inputs: Union[Tensor, List[Tensor]],
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if isinstance(inputs, torch.Tensor):
            modality_tensors = [inputs]
        elif isinstance(inputs, (list, tuple)):
            modality_tensors = list(inputs)
        else:
            raise TypeError("inputs must be a Tensor or a list/tuple of Tensors.")

        encoder_pyramids: List[Dict[str, Tensor]] = []
        for tensor in modality_tensors:
            if tensor.dim() != 4:
                raise ValueError("Expected 4D tensor [B, C, H, W] for each modality.")
            if self.image_adapter is not None:
                tensor = self.image_adapter(tensor)
            final_map, pyramid = self.encoder(tensor, return_intermediate=True)
            pyramid = dict(pyramid)  # shallow copy
            pyramid["layer_last"] = final_map
            encoder_pyramids.append(pyramid)

        fused_pyramid = self.fusion(encoder_pyramids)
        decoded = self.decoder(fused_pyramid)
        logits = self.segmentation_head(decoded)

        if return_features:
            debug_info = {
                "encoder_pyramids": encoder_pyramids,
                "fused": fused_pyramid,
                "decoded": decoded,
            }
            return logits, debug_info
        return logits


__all__ = [
    "DINOv3Encoder",
    "MultiModalFusion",
    "FeaturePyramidDecoder",
    "CombinedLoss",
    "DINOv3MultiModalSeg",
]
