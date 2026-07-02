"""
DINOv3-PhysioMamba Interactive Segmentation Model

统一架构支持:
1. 参考帧引导 (Semi-supervised VOS / 3D propagation)
2. 点击提示 (Interactive segmentation)
3. 固定类别 (Traditional segmentation)

核心创新:
- DINOv3 提供强大的语义特征 (跨帧匹配)
- PhysioMamba 提供高效的时序建模 (替代 Memory Attention)
- 统一医学 3D 和视频分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.frozen_dinov3_multi_modal_seg import FrozenDINOv3Encoder, FeaturePyramidDecoder
from model.mamba3.mimo_adapter import PhysioMambaBlock
from .reference_encoder import ReferenceEncoder, ClickEncoder
from .cross_frame_matching import CrossFrameMatching, DINOFeatureMatcher, MemoryBank


class DINOv3PhysioMambaInteractive(nn.Module):
    """
    交互式 DINOv3-PhysioMamba 分割模型
    
    架构:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input Frames ──→ DINOv3 Encoder ──→ Frame Features         │
    │                                           │                 │
    │  Reference/Prompt ──→ Reference Encoder ──┤                 │
    │                                           ▼                 │
    │                              Cross-Frame Matching           │
    │                                           │                 │
    │                                           ▼                 │
    │                              PhysioMamba Memory             │
    │                                           │                 │
    │                                           ▼                 │
    │                              FPN Decoder ──→ Masks          │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        # Encoder config
        model_name: str = 'dinov3_vitb16',
        pretrained_weights: Optional[str] = None,
        unfreeze_last_n_blocks: int = 4,
        # Adapter config
        d_state: int = 16,
        dt_rank: int = 8,
        expand: int = 2,
        # Interactive config
        num_target_tokens: int = 64,
        num_matching_layers: int = 2,
        # Decoder config
        decoder_channels: Tuple[int, ...] = (256, 128, 64),
        # Output config
        num_classes: Optional[int] = None,  # None = binary (interactive)
        # Ablation config
        use_mamba: bool = True,
        use_parallel_scan: bool = False,
        use_metric_positions: bool = True,
        # RoPE config (新增)
        rope_type: str = "segmentation",
        rope_base_temporal: float = 50000.0,
        rope_base_spatial: float = 10000.0,
        rope_temporal_ratio: float = 0.25,
    ):
        super().__init__()
        
        self.d_model = 768  # ViT-B
        self.num_classes = num_classes
        self.use_mamba = use_mamba
        self.use_parallel_scan = use_parallel_scan
        self.use_metric_positions = use_metric_positions
        
        # ==================== Encoder ====================
        print("[Debug] Initializing FrozenDINOv3Encoder...", flush=True)
        self.encoder = FrozenDINOv3Encoder(
            model_name=model_name,
            pretrained_weights=pretrained_weights,
            freeze=True,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        self.d_model = self.encoder.embed_dim
        print(f"[Debug] Encoder initialized. d_model={self.d_model}", flush=True)
        
        # ==================== Interactive Modules ====================
        # Reference encoder (for reference-guided mode)
        print("[Debug] Initializing ReferenceEncoder...", flush=True)
        self.reference_encoder = ReferenceEncoder(
            d_model=self.d_model,
            num_target_tokens=num_target_tokens,
        )
        
        # Click encoder (for click prompt mode)
        print("[Debug] Initializing ClickEncoder...", flush=True)
        self.click_encoder = ClickEncoder(
            d_model=self.d_model,
        )
        
        # Cross-frame matching
        print("[Debug] Initializing CrossFrameMatching...", flush=True)
        self.cross_matching = CrossFrameMatching(
            d_model=self.d_model,
            n_heads=8,
            num_layers=num_matching_layers,
        )
        
        # Direct feature matcher (simple baseline)
        self.feature_matcher = DINOFeatureMatcher(
            d_model=self.d_model,
        )
        
        # Memory bank (for multi-frame propagation)
        self.memory_bank = MemoryBank(
            d_model=self.d_model,
            max_memories=8,
        )
        
        # ==================== Temporal Modeling ====================
        # PhysioMamba for temporal consistency
        print(f"[Debug] Initializing PhysioMambaBlock (type={rope_type})...", flush=True)
        self.mamba_temporal = PhysioMambaBlock(
            d_model=self.d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            expand=expand,
            use_parallel_scan=use_parallel_scan,
            # RoPE配置
            rope_type=rope_type,
            rope_base_temporal=rope_base_temporal,
            rope_base_spatial=rope_base_spatial,
            rope_temporal_ratio=rope_temporal_ratio,
            rope_num_axes=3,  # Unified Metric Space-Time: (t, h, w)
        )
        print("[Debug] PhysioMambaBlock initialized.", flush=True)
        
        # ==================== Decoder ====================
        # 简单的卷积解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(self.d_model, decoder_channels[0], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling path
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels[i], decoder_channels[i+1], 3, padding=1),
                nn.BatchNorm2d(decoder_channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            )
            for i in range(len(decoder_channels) - 1)
        ])
        
        # ==================== Output Head ====================
        if num_classes is None:
            # Interactive mode: binary segmentation
            self.head = nn.Conv2d(decoder_channels[-1], 1, 1)
            self.output_channels = 1
        else:
            # Fixed-class mode
            self.head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
            self.output_channels = num_classes
        
        # Occlusion prediction (object visible or not)
        self.occlusion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def _unfreeze_encoder_blocks(self, n_blocks: int):
        """解冻 encoder 最后 N 个 blocks"""
        if n_blocks <= 0:
            return
        
        # 假设 encoder 有 blocks 属性
        if hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'blocks'):
            blocks = self.encoder.model.blocks
            total_blocks = len(blocks)
            for i in range(total_blocks - n_blocks, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
    
    def encode_reference(
        self,
        frame: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper to encode a reference frame/mask pair into tokens for Memory Bank.
        
        Args:
            frame: [B, 3, H, W]
            mask: [B, 1, H, W]
            
        Returns:
            target_tokens: [B, N, D]
            global_repr: [B, D]
        """
        # Encode reference frame
        ref_features = self.encoder(frame)  # [B, D, h, w]
        ref_features = ref_features.permute(0, 2, 3, 1)  # [B, h, w, D]
        
        # Get target representation
        target_tokens, global_repr = self.reference_encoder(
            ref_features, mask
        )
        return target_tokens, global_repr

    def forward(
        self,
        frames: torch.Tensor,
        reference_frame: Optional[torch.Tensor] = None,
        reference_mask: Optional[torch.Tensor] = None,
        # New args for multi-reference inference
        reference_tokens: Optional[torch.Tensor] = None,
        reference_global: Optional[torch.Tensor] = None,
        
        click_points: Optional[torch.Tensor] = None,
        click_labels: Optional[torch.Tensor] = None,
        spacing: Optional[torch.Tensor] = None,
        use_memory: bool = False,
        return_intermediate: bool = False,
        mamba_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_mamba_state: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            frames: [B, T, C, H, W] 输入帧序列
            reference_frame: [B, C, H, W] 参考帧 (optional)
            reference_mask: [B, 1, H, W] 参考帧 mask (optional)
            reference_tokens: [B, N, D] 预计算的参考 tokens (optional, overrides reference_frame)
            reference_global: [B, D] 预计算的全局参考 (optional)
            spacing: [B, T] or [B, T, num_axes] 物理间距 (optional, for PhysioMamba)
            click_points: [B, N, 2] 点击坐标 (optional)
            click_labels: [B, N] 点击标签 (optional)
            use_memory: 是否使用 memory bank
            return_intermediate: 是否返回中间结果
        
        Returns:
            output_dict: {
                'masks': [B, T, C, H, W] 预测 masks
                'confidence': [B, T, H, W] 匹配置信度
                'occlusion': [B, T] 遮挡预测
                ... (intermediate results if requested)
            }
        """
        B, T, C, H, W = frames.shape
        device = frames.device
        
        output_dict = {}
        
        # ========== 1. Encode all frames ==========
        all_frames = frames.reshape(B * T, C, H, W)
        frame_features = self.encoder(all_frames)  # [B*T, D, h, w]
        _, D, h, w = frame_features.shape
        # Reshape to [B, T, h, w, D]
        frame_features = frame_features.permute(0, 2, 3, 1)  # [B*T, h, w, D]
        frame_features = frame_features.reshape(B, T, h, w, D)
        
        # ========== 2. Reference-guided mode ==========
        # Priority: reference_tokens > reference_frame > click_points
        
        if reference_tokens is not None:
             # Use provided tokens (e.g. from Memory Bank)
             target_tokens = reference_tokens
             # global_repr is optional or provided
             global_repr = reference_global
             
             # Cross-frame matching
             matched_features, match_confidence = self.cross_matching(
                frame_features, target_tokens, global_repr
             )
             
        elif reference_frame is not None and reference_mask is not None:
            # Encode reference frame
            ref_features = self.encoder(reference_frame)  # [B, D, h, w]
            ref_features = ref_features.permute(0, 2, 3, 1)  # [B, h, w, D]
            
            # Get target representation
            target_tokens, global_repr = self.reference_encoder(
                ref_features, reference_mask
            )
            
            # Cross-frame matching
            matched_features, match_confidence = self.cross_matching(
                frame_features, target_tokens, global_repr
            )
            
            if return_intermediate:
                output_dict['target_tokens'] = target_tokens
                output_dict['global_repr'] = global_repr
                output_dict['match_confidence'] = match_confidence
        
        # ========== 3. Click prompt mode ==========
        elif click_points is not None and click_labels is not None:
            # Encode clicks
            click_tokens = self.click_encoder(
                click_points, click_labels, image_size=(H, W)
            )  # [B, N, D]
            
            # Use first frame for reference
            first_frame_features = frame_features[:, 0]  # [B, h, w, D]
            
            # Get target representation from clicks
            target_tokens, global_repr = self._clicks_to_target(
                first_frame_features, click_points, click_labels, click_tokens
            )
            
            # Cross-frame matching
            matched_features, match_confidence = self.cross_matching(
                frame_features, target_tokens, global_repr
            )
        
        # ========== 4. Fixed-class mode (no interaction) ==========
        else:
            matched_features = frame_features
            match_confidence = torch.ones(B, T, h, w, device=device)
        
        # ========== 5. Memory bank (optional) ==========
        if use_memory and self.memory_bank.memory_bank is not None:
            memory_features, attn_weights = self.memory_bank.read_memory(
                matched_features.reshape(B * T, h, w, D)
            )
            memory_features = memory_features.reshape(B, T, h, w, D)
            matched_features = matched_features + 0.5 * memory_features
        
        # ========== 6. Temporal modeling with PhysioMamba ==========
        if self.use_mamba:
            # Reshape for Mamba: [B*h*w, T, D]
            temporal_input = matched_features.permute(0, 2, 3, 1, 4)  # [B, h, w, T, D]
            temporal_input = temporal_input.reshape(B * h * w, T, D)
            
            # Prepare spacing and positions for Metric-Aware Modeling
            mamba_spacing = None
            mamba_positions = None
            
            if spacing is not None:
                # -------------------------------------------------------
                # 1. Spacing Preparation (for dt calculation & RoPE scaling)
                # -------------------------------------------------------
                # spacing: [B, T] -> [B*h*w, T]
                if spacing.dim() == 2:
                    # spacing is delta_t [B, T]
                    # Broadcast to spatial pixels
                    mamba_spacing = spacing.unsqueeze(1).unsqueeze(1) # [B, 1, 1, T]
                    mamba_spacing = mamba_spacing.expand(-1, h, w, -1) # [B, h, w, T]
                    mamba_spacing = mamba_spacing.reshape(B * h * w, T)
                    
                    # -------------------------------------------------------
                    # 2. Position Construction (Unified Metric Space-Time)
                    # -------------------------------------------------------
                    # Construct [t_metric, h_idx, w_idx]
                    
                    # A. Temporal Metric Position
                    # t_pos = cumsum(delta_t)
                    if self.use_metric_positions:
                        t_metric = torch.cumsum(spacing, dim=1) # [B, T]
                        t_metric = t_metric - t_metric[:, :1]   # Start from 0
                    else:
                        t_metric = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
                    
                    # Broadcast t_metric to [B, h, w, T]
                    t_pos = t_metric.unsqueeze(1).unsqueeze(1).expand(-1, h, w, -1) # [B, h, w, T]
                    
                    # B. Spatial Index Positions
                    # Generate h, w indices
                    y_grid = torch.arange(h, device=device).float()
                    x_grid = torch.arange(w, device=device).float()
                    mesh_y, mesh_x = torch.meshgrid(y_grid, x_grid, indexing='ij') # [h, w]
                    
                    # Expand to [B, h, w, T]
                    # Note: Spatial position is constant across time T for a single pixel "tube"
                    h_pos = mesh_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, T)
                    w_pos = mesh_x.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, T)
                    
                    # C. Stack to [B*h*w, T, 3]
                    # Flatten spatial dims first
                    t_pos_flat = t_pos.reshape(B * h * w, T)
                    h_pos_flat = h_pos.reshape(B * h * w, T)
                    w_pos_flat = w_pos.reshape(B * h * w, T)
                    
                    mamba_positions = torch.stack([t_pos_flat, h_pos_flat, w_pos_flat], dim=-1)
                    
                elif spacing.dim() == 3: # [B, T, num_axes]
                    # Advanced case: multi-axis spacing provided
                    # Flatten spatial dim but keep axes dim
                    mamba_spacing = spacing.unsqueeze(1).unsqueeze(1) # [B, 1, 1, T, axes]
                    mamba_spacing = mamba_spacing.expand(-1, h, w, -1, -1)
                    mamba_spacing = mamba_spacing.reshape(B * h * w, T, -1)
                    
                    # TODO: If spacing has 3 axes (dt, dy, dx), we could do cumsum on all of them?
                    # For now, fallback to t_metric + index spatial (safest baseline)
                    # Or extract t from spacing[..., 0]
                    delta_t = spacing[..., 0]
                    if self.use_metric_positions:
                        t_metric = torch.cumsum(delta_t, dim=1)
                        t_metric = t_metric - t_metric[:, :1]
                    else:
                        t_metric = torch.arange(T, device=device).float().unsqueeze(0).expand(B, -1)
                    t_pos = t_metric.unsqueeze(1).unsqueeze(1).expand(-1, h, w, -1)
                    
                    y_grid = torch.arange(h, device=device).float()
                    x_grid = torch.arange(w, device=device).float()
                    mesh_y, mesh_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
                    h_pos = mesh_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, T)
                    w_pos = mesh_x.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, T)
                    
                    t_pos_flat = t_pos.reshape(B * h * w, T)
                    h_pos_flat = h_pos.reshape(B * h * w, T)
                    w_pos_flat = w_pos.reshape(B * h * w, T)
                    
                    mamba_positions = torch.stack([t_pos_flat, h_pos_flat, w_pos_flat], dim=-1)

            # Apply Mamba with Metric-Awareness
            need_mamba_state = return_mamba_state or (mamba_state is not None)
            temporal_output = self.mamba_temporal(
                temporal_input, 
                spacing=mamba_spacing,
                positions=mamba_positions, # Pass explicit metric coordinates
                state=mamba_state,
                return_state=need_mamba_state,
            )
            if need_mamba_state:
                temporal_output, mamba_state = temporal_output
                output_dict['mamba_state'] = mamba_state
            
            # Reshape back: [B, T, h, w, D]
            temporal_output = temporal_output.reshape(B, h, w, T, D)
            temporal_output = temporal_output.permute(0, 3, 1, 2, 4)
        else:
            # Ablation: No temporal modeling
            temporal_output = matched_features
        
        # ========== 7. Decode ==========
        # [B, T, h, w, D] -> [B*T, D, h, w]
        decoder_input = temporal_output.reshape(B * T, h, w, D)
        decoder_input = decoder_input.permute(0, 3, 1, 2).contiguous()
        
        # Decoder
        decoded = self.decoder(decoder_input)
        
        # Upsample
        for upsample in self.upsample_layers:
            decoded = upsample(decoded)
        
        # ========== 8. Output heads ==========
        # Segmentation masks
        logits = self.head(decoded)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=False)
        logits = logits.reshape(B, T, self.output_channels, H, W)
        
        # Occlusion prediction
        occlusion = self.occlusion_head(decoded)
        occlusion = occlusion.reshape(B, T)
        
        output_dict['masks'] = logits
        output_dict['occlusion'] = occlusion
        output_dict['confidence'] = F.interpolate(
            match_confidence.reshape(B * T, 1, h, w),
            (H, W), mode='bilinear', align_corners=False
        ).reshape(B, T, H, W)
        
        return output_dict
    
    def _clicks_to_target(
        self,
        frame_features: torch.Tensor,
        click_points: torch.Tensor,
        click_labels: torch.Tensor,
        click_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从点击生成目标表示
        
        Args:
            frame_features: [B, h, w, D]
            click_points: [B, N, 2]
            click_labels: [B, N]
            click_tokens: [B, N, D]
        
        Returns:
            target_tokens: [B, N, D]
            global_repr: [B, D]
        """
        B, h, w, D = frame_features.shape
        N = click_points.shape[1]
        
        # 从点击位置提取 DINOv3 特征
        # 归一化坐标到 feature map
        coords_normalized = click_points.clone()
        coords_normalized[..., 0] = coords_normalized[..., 0] / (w * 14) * 2 - 1
        coords_normalized[..., 1] = coords_normalized[..., 1] / (h * 14) * 2 - 1
        
        # Grid sample
        features_chw = frame_features.permute(0, 3, 1, 2)  # [B, D, h, w]
        grid = coords_normalized.reshape(B, N, 1, 2)  # [B, N, 1, 2]
        sampled = F.grid_sample(
            features_chw, grid, mode='bilinear', align_corners=False
        )  # [B, D, N, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, D]
        
        # 结合 click tokens
        target_tokens = sampled + click_tokens
        
        # 正点击的平均作为全局表示
        pos_mask = (click_labels > 0.5).unsqueeze(-1).float()
        num_pos = pos_mask.sum(dim=1).clamp(min=1)
        global_repr = (target_tokens * pos_mask).sum(dim=1) / num_pos
        
        return target_tokens, global_repr
    
    def propagate_mask(
        self,
        video: torch.Tensor,
        reference_frame: torch.Tensor,
        reference_mask: torch.Tensor,
        bidirectional: bool = True,
    ) -> torch.Tensor:
        """
        传播 mask 到整个视频
        
        Args:
            video: [B, T, C, H, W]
            reference_frame: [B, C, H, W]
            reference_mask: [B, 1, H, W]
            bidirectional: 是否双向传播
        
        Returns:
            masks: [B, T, 1, H, W]
        """
        B, T, C, H, W = video.shape
        
        # 单向传播
        output = self.forward(
            video,
            reference_frame=reference_frame,
            reference_mask=reference_mask,
        )
        forward_masks = output['masks']
        
        if not bidirectional:
            return torch.sigmoid(forward_masks)
        
        # 反向传播
        video_reversed = video.flip(dims=[1])
        output_reversed = self.forward(
            video_reversed,
            reference_frame=reference_frame,
            reference_mask=reference_mask,
        )
        backward_masks = output_reversed['masks'].flip(dims=[1])
        
        # 融合
        combined_masks = (forward_masks + backward_masks) / 2
        
        return torch.sigmoid(combined_masks)
    
    def interactive_refine(
        self,
        video: torch.Tensor,
        initial_masks: torch.Tensor,
        correction_frame_idx: int,
        correction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        交互式修正
        
        用户在某一帧修正 mask 后，重新传播
        
        Args:
            video: [B, T, C, H, W]
            initial_masks: [B, T, 1, H, W] 初始预测
            correction_frame_idx: 修正帧的索引
            correction_mask: [B, 1, H, W] 修正后的 mask
        
        Returns:
            refined_masks: [B, T, 1, H, W]
        """
        B, T, C, H, W = video.shape
        
        # 使用修正帧作为新的参考帧
        reference_frame = video[:, correction_frame_idx]
        
        # 重新传播
        refined_masks = self.propagate_mask(
            video,
            reference_frame=reference_frame,
            reference_mask=correction_mask,
            bidirectional=True,
        )
        
        # 在修正帧使用用户提供的 mask
        refined_masks[:, correction_frame_idx] = torch.sigmoid(correction_mask)
        
        return refined_masks
    
    def reset_memory(self):
        """重置 memory bank"""
        self.memory_bank.reset()


def build_interactive_model(
    config: Dict[str, Any],
) -> DINOv3PhysioMambaInteractive:
    """
    从配置构建模型
    
    Args:
        config: 模型配置字典
        
    RoPE 配置 (三种类型):
        - "standard": 标准 ComplexRoPE (base=10000)
        - "physio_v1": PhysioRoPE (多轴分块分配)
        - "segmentation": SegmentationMRoPE (时空解耦，推荐)
    
    Returns:
        model: DINOv3PhysioMambaInteractive
    """
    return DINOv3PhysioMambaInteractive(
        model_name=config.get('model_name', 'dinov3_vitb16'),
        pretrained_weights=config.get('pretrained_weights', None),
        unfreeze_last_n_blocks=config.get('unfreeze_last_n_blocks', 4),
        d_state=config.get('d_state', 16),
        dt_rank=config.get('dt_rank', 8),
        expand=config.get('expand', 2),
        num_target_tokens=config.get('num_target_tokens', 64),
        num_matching_layers=config.get('num_matching_layers', 2),
        decoder_channels=tuple(config.get('decoder_channels', [256, 128, 64])),
        num_classes=config.get('num_classes', None),
        use_mamba=config.get('use_mamba', True),
        use_parallel_scan=config.get('use_parallel_scan', False),
        use_metric_positions=config.get('use_metric_positions', True),
        # RoPE 配置
        rope_type=config.get('rope_type', 'standard'),
        rope_base_temporal=config.get('rope_base_temporal', 50000.0),
        rope_base_spatial=config.get('rope_base_spatial', 10000.0),
        rope_temporal_ratio=config.get('rope_temporal_ratio', 0.25),
    )
