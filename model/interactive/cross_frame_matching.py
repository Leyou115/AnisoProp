"""
Cross-Frame Feature Matching

核心思想:
- DINOv3 特征天然具有语义对应能力
- 通过 cross-attention 将参考帧目标信息传递到目标帧
- 生成匹配增强的特征用于分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossFrameMatching(nn.Module):
    """
    跨帧特征匹配模块
    
    Query: 目标帧的 DINOv3 特征
    Key/Value: 参考帧的目标表示
    
    输出: 匹配增强的目标帧特征
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 特征投影
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 匹配置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # 输出融合
        self.output_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(
        self,
        query_features: torch.Tensor,
        reference_tokens: torch.Tensor,
        reference_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_features: [B, T, H, W, D] 目标帧的 DINOv3 特征
            reference_tokens: [B, N, D] 参考帧的目标表示 tokens
            reference_global: [B, D] 参考帧的全局表示 (可选)
        
        Returns:
            matched_features: [B, T, H, W, D] 匹配增强的特征
            match_confidence: [B, T, H, W] 匹配置信度图
        """
        B, T, H, W, D = query_features.shape
        N = reference_tokens.shape[1]
        
        # Flatten spatial dimensions
        query_flat = query_features.view(B * T, H * W, D)  # [BT, HW, D]
        
        # Expand reference for each frame
        ref_expanded = reference_tokens.unsqueeze(1).expand(-1, T, -1, -1)
        ref_flat = ref_expanded.reshape(B * T, N, D)  # [BT, N, D]
        
        # Project
        Q = self.query_proj(query_flat)
        K = self.key_proj(ref_flat)
        V = self.value_proj(ref_flat)
        
        # Cross-attention layers
        matched = Q
        for cross_attn in self.cross_attn_layers:
            matched = cross_attn(matched, K, V)
        
        # 融合原始特征和匹配特征
        fused = self.output_fusion(
            torch.cat([query_flat, matched], dim=-1)
        )
        
        # 计算匹配置信度
        confidence = self.confidence_head(matched)  # [BT, HW, 1]
        
        # 如果有全局表示，用作额外的调制
        if reference_global is not None:
            global_expanded = reference_global.unsqueeze(1).expand(-1, T, -1)
            global_flat = global_expanded.reshape(B * T, 1, D)
            
            # 计算与全局表示的相似度作为额外置信度
            global_sim = F.cosine_similarity(
                matched.mean(dim=1, keepdim=True),
                global_flat,
                dim=-1
            ).unsqueeze(-1)  # [BT, 1, 1]
            
            confidence = confidence * (0.5 + 0.5 * global_sim)
        
        # Reshape back
        matched_features = fused.view(B, T, H, W, D)
        match_confidence = confidence.view(B, T, H, W)
        
        return matched_features, match_confidence


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with residual connection
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, L_q, D]
            key: [B, L_k, D]
            value: [B, L_k, D]
        
        Returns:
            output: [B, L_q, D]
        """
        # Cross-attention with residual
        attn_out, _ = self.cross_attn(query, key, value)
        query = self.norm1(query + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(query)
        output = self.norm2(query + ffn_out)
        
        return output


class DINOFeatureMatcher(nn.Module):
    """
    直接利用 DINOv3 特征进行像素级匹配
    
    核心: DINOv3 的 patch tokens 天然支持语义对应
    """
    
    def __init__(self, d_model: int = 768, temperature: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        
        # 可选: 特征精炼 (类似 DINO-Tracker 的 Delta-DINO)
        self.feature_refiner = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 是否使用精炼
        self.use_refiner = True
    
    def forward(
        self,
        query_features: torch.Tensor,
        reference_features: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        直接特征匹配 (不需要学习的 cross-attention)
        
        Args:
            query_features: [B, H, W, D] 目标帧 DINOv3 特征
            reference_features: [B, H, W, D] 参考帧 DINOv3 特征
            reference_mask: [B, 1, H, W] 参考帧 mask
        
        Returns:
            similarity_map: [B, H, W] 与参考目标的相似度
            propagated_mask: [B, 1, H, W] 传播的 mask
        """
        B, H, W, D = query_features.shape
        
        # 可选精炼
        if self.use_refiner:
            query_features = query_features + self.feature_refiner(query_features)
            reference_features = reference_features + self.feature_refiner(reference_features)
        
        # Normalize features
        query_norm = F.normalize(query_features, dim=-1)
        ref_norm = F.normalize(reference_features, dim=-1)
        
        # Flatten
        query_flat = query_norm.view(B, H * W, D)  # [B, HW, D]
        ref_flat = ref_norm.view(B, H * W, D)  # [B, HW, D]
        
        # Resize mask
        mask_small = F.interpolate(
            reference_mask.float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        mask_flat = mask_small.view(B, H * W)  # [B, HW]
        
        # 计算参考目标的平均特征
        masked_ref = ref_flat * mask_flat.unsqueeze(-1)
        target_feat = masked_ref.sum(dim=1) / (mask_flat.sum(dim=1, keepdim=True) + 1e-6)
        target_feat = F.normalize(target_feat, dim=-1)  # [B, D]
        
        # 计算相似度
        similarity = torch.einsum('bnd,bd->bn', query_flat, target_feat)  # [B, HW]
        similarity = similarity / self.temperature
        similarity_map = similarity.view(B, H, W)
        
        # Soft mask propagation
        propagated_mask = torch.sigmoid(similarity_map).unsqueeze(1)  # [B, 1, H, W]
        
        # 上采样
        propagated_mask = F.interpolate(
            propagated_mask,
            size=(H * 14, W * 14),  # 假设 patch size = 14
            mode='bilinear',
            align_corners=False
        )
        
        return similarity_map, propagated_mask


class MemoryBank(nn.Module):
    """
    Memory Bank for storing reference frame information
    
    类似 SAM2 / XMem 的 memory bank
    """
    
    def __init__(
        self,
        d_model: int = 768,
        max_memories: int = 8,
        memory_dim: int = 64,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_memories = max_memories
        self.memory_dim = memory_dim
        
        # Memory 压缩
        self.memory_proj = nn.Linear(d_model, memory_dim)
        self.memory_unproj = nn.Linear(memory_dim, d_model)
        
        # Memory 存储 (运行时填充)
        self.register_buffer('memory_bank', None)
        self.register_buffer('memory_masks', None)
        self.memory_count = 0
    
    def reset(self):
        """重置 memory bank"""
        self.memory_bank = None
        self.memory_masks = None
        self.memory_count = 0
    
    def add_memory(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        添加新的 memory
        
        Args:
            features: [B, H, W, D] 帧特征
            mask: [B, 1, H, W] 预测的 mask
        """
        B, H, W, D = features.shape
        
        # 压缩特征
        compressed = self.memory_proj(features)  # [B, H, W, memory_dim]
        
        # 结合 mask 信息
        mask_small = F.interpolate(mask.float(), (H, W), mode='bilinear')
        compressed = compressed * (0.5 + 0.5 * mask_small.permute(0, 2, 3, 1))
        
        if self.memory_bank is None:
            self.memory_bank = compressed.unsqueeze(1)
            self.memory_masks = mask_small.unsqueeze(1)
        else:
            self.memory_bank = torch.cat([
                self.memory_bank, compressed.unsqueeze(1)
            ], dim=1)
            self.memory_masks = torch.cat([
                self.memory_masks, mask_small.unsqueeze(1)
            ], dim=1)
        
        self.memory_count += 1
        
        # 如果超过最大数量，移除最旧的
        if self.memory_count > self.max_memories:
            self.memory_bank = self.memory_bank[:, 1:]
            self.memory_masks = self.memory_masks[:, 1:]
            self.memory_count = self.max_memories
    
    def read_memory(
        self,
        query_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取 memory
        
        Args:
            query_features: [B, H, W, D] 当前帧特征
        
        Returns:
            memory_features: [B, H, W, D] 聚合的 memory 特征
            attention_weights: [B, M, H, W] 各 memory 的权重
        """
        if self.memory_bank is None:
            return query_features, None
        
        B, H, W, D = query_features.shape
        M = self.memory_bank.shape[1]
        
        # 压缩 query
        query_compressed = self.memory_proj(query_features)  # [B, H, W, memory_dim]
        
        # 计算与每个 memory 的相似度
        query_flat = query_compressed.view(B, H * W, -1)  # [B, HW, memory_dim]
        memory_flat = self.memory_bank.view(B, M, H * W, -1)  # [B, M, HW, memory_dim]
        
        # Attention
        similarity = torch.einsum('bqd,bmkd->bmqk', query_flat, memory_flat)
        similarity = similarity / (self.memory_dim ** 0.5)
        attention = F.softmax(similarity, dim=-1)  # [B, M, HW, HW]
        
        # 聚合
        memory_values = self.memory_unproj(
            self.memory_bank.view(B * M, H, W, -1)
        ).view(B, M, H * W, D)
        
        aggregated = torch.einsum('bmqk,bmkd->bqd', attention, memory_values)
        aggregated = aggregated.view(B, H, W, D)
        
        # 简化的 attention weights
        attention_weights = attention.mean(dim=-1).view(B, M, H, W)
        
        return aggregated, attention_weights
