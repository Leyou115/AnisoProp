"""
Reference Encoder for Interactive Segmentation

核心思想:
- 利用 DINOv3 的语义特征
- 通过 mask 提取目标表示
- 支持全局 (pooled) 和局部 (tokens) 表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ReferenceEncoder(nn.Module):
    """
    编码参考帧 + mask，生成目标表示
    
    输入:
    - DINOv3 特征 (参考帧)
    - 用户提供的 mask
    
    输出:
    - target_tokens: 前景区域的特征 tokens
    - global_repr: 全局目标表示 (pooled)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_target_tokens: int = 64,
        use_mask_embedding: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_target_tokens = num_target_tokens
        
        # Mask embedding (可学习的 mask 位置编码)
        self.use_mask_embedding = use_mask_embedding
        if use_mask_embedding:
            self.mask_conv = nn.Sequential(
                nn.Conv2d(1, 64, 7, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, d_model, 1),
            )
        
        # Target token 聚合
        self.target_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # 可学习的 query tokens (用于固定数量输出)
        self.query_tokens = nn.Parameter(torch.randn(1, num_target_tokens, d_model))
        
        # 全局表示投影
        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(
        self,
        dino_features: torch.Tensor,
        mask: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dino_features: [B, H, W, D] DINOv3 特征
            mask: [B, 1, H_orig, W_orig] 用户提供的 mask
            return_all_tokens: 是否返回所有前景 tokens
        
        Returns:
            target_tokens: [B, num_target_tokens, D] 目标表示 tokens
            global_repr: [B, D] 全局目标表示
        """
        B, H, W, D = dino_features.shape
        
        # Resize mask to feature size
        mask_small = F.interpolate(
            mask.float(), 
            size=(H, W), 
            mode='bilinear',
            align_corners=False
        )  # [B, 1, H, W]
        
        # 二值化
        mask_binary = (mask_small > 0.5).float()
        
        # 1. 提取前景特征
        features_flat = dino_features.view(B, H * W, D)  # [B, HW, D]
        mask_flat = mask_binary.view(B, H * W)  # [B, HW]
        
        # Masked features (背景置零)
        masked_features = features_flat * mask_flat.unsqueeze(-1)
        
        # 2. 添加 mask embedding (可选)
        if self.use_mask_embedding:
            mask_embed = self.mask_conv(mask_small)  # [B, D, H, W]
            mask_embed = mask_embed.permute(0, 2, 3, 1).view(B, H * W, D)
            masked_features = masked_features + mask_embed * mask_flat.unsqueeze(-1)
        
        # 3. 全局表示 (masked average pooling)
        num_foreground = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
        global_repr = masked_features.sum(dim=1) / num_foreground  # [B, D]
        global_repr = self.global_proj(global_repr)
        
        # 4. Target tokens (通过 cross-attention 聚合)
        # 使用可学习的 query tokens 从前景特征中提取信息
        query = self.query_tokens.expand(B, -1, -1)  # [B, num_tokens, D]
        
        # 只关注前景区域 (通过 attention mask)
        # 注意: PyTorch attention mask: True = 忽略
        attn_mask = (mask_flat < 0.5)  # [B, HW], True = 背景
        
        # 将 query 和 masked_features concat，然后 self-attention
        combined = torch.cat([query, masked_features], dim=1)  # [B, num_tokens + HW, D]
        
        # 创建 attention mask
        # query 可以 attend 到前景，前景只能 attend 到自己
        seq_len = combined.shape[1]
        full_mask = torch.zeros(B, seq_len, seq_len, device=combined.device, dtype=torch.bool)
        # query -> features: mask out background
        full_mask[:, :self.num_target_tokens, self.num_target_tokens:] = attn_mask.unsqueeze(1)
        
        # Transformer 聚合 (简化版本，不使用复杂 mask)
        aggregated = self.target_aggregator(combined)
        
        # 取 query 部分作为 target tokens
        target_tokens = aggregated[:, :self.num_target_tokens]  # [B, num_tokens, D]
        
        if return_all_tokens:
            # 返回所有前景 tokens (用于更精细的匹配)
            foreground_tokens = self._extract_foreground_tokens(
                features_flat, mask_flat
            )
            return target_tokens, global_repr, foreground_tokens
        
        return target_tokens, global_repr
    
    def _extract_foreground_tokens(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        max_tokens: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取前景区域的所有 tokens
        
        Args:
            features: [B, HW, D]
            mask: [B, HW]
            max_tokens: 最大 token 数量
        
        Returns:
            foreground_tokens: [B, max_tokens, D] (padding if needed)
            valid_mask: [B, max_tokens] 有效位置
        """
        B, HW, D = features.shape
        device = features.device
        
        foreground_tokens = torch.zeros(B, max_tokens, D, device=device)
        valid_mask = torch.zeros(B, max_tokens, device=device, dtype=torch.bool)
        
        for b in range(B):
            fg_indices = torch.where(mask[b] > 0.5)[0]
            num_fg = min(len(fg_indices), max_tokens)
            
            if num_fg > 0:
                # 随机采样或全部保留
                if len(fg_indices) > max_tokens:
                    perm = torch.randperm(len(fg_indices), device=device)[:max_tokens]
                    fg_indices = fg_indices[perm]
                
                foreground_tokens[b, :num_fg] = features[b, fg_indices[:num_fg]]
                valid_mask[b, :num_fg] = True
        
        return foreground_tokens, valid_mask


class ClickEncoder(nn.Module):
    """
    编码点击提示 (正/负点击)
    
    类似 SAM 的 prompt encoder
    """
    
    def __init__(self, d_model: int = 768, max_clicks: int = 16):
        super().__init__()
        
        self.d_model = d_model
        self.max_clicks = max_clicks
        
        # 位置编码 (从坐标生成)
        self.coord_mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        
        # 正/负点击 embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.neg_embed = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 点击聚合
        self.click_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 2,
                batch_first=True,
            ),
            num_layers=2,
        )
    
    def forward(
        self,
        click_points: torch.Tensor,
        click_labels: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """
        Args:
            click_points: [B, N, 2] 点击坐标 (x, y)，归一化到 [0, 1]
            click_labels: [B, N] 点击标签 (1=正, 0=负)
            image_size: (H, W) 图像尺寸
        
        Returns:
            click_tokens: [B, N, D] 点击 token 表示
        """
        B, N, _ = click_points.shape
        
        # 归一化坐标
        coords_normalized = click_points.clone()
        coords_normalized[..., 0] = coords_normalized[..., 0] / image_size[1]
        coords_normalized[..., 1] = coords_normalized[..., 1] / image_size[0]
        
        # 位置编码
        pos_encoding = self.coord_mlp(coords_normalized)  # [B, N, D]
        
        # 添加正/负 embedding
        label_embed = torch.where(
            click_labels.unsqueeze(-1) > 0.5,
            self.pos_embed.expand(B, N, -1),
            self.neg_embed.expand(B, N, -1),
        )
        
        click_tokens = pos_encoding + label_embed
        
        # 聚合
        click_tokens = self.click_aggregator(click_tokens)
        
        return click_tokens
