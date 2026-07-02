import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .physio_mamba_primitives import (
    MetricAwareTrapezoidalParameters, 
    ComplexRoPE,
    PhysioRoPE,
    SegmentationMRoPE,
)

# 检查 CUDA 优化是否可用
MAMBA_CUDA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    MAMBA_CUDA_AVAILABLE = True
except ImportError:
    pass

# 检查 causal_conv1d 是否可用
CAUSAL_CONV1D_AVAILABLE = False
try:
    from causal_conv1d import causal_conv1d_fn
    CAUSAL_CONV1D_AVAILABLE = True
except ImportError:
    pass

class PhysioMambaBlock(nn.Module):
    """
    Physio-Mamba Block with Multi-Scale Input and Metric-Aware Z-Scanning.
    
    支持多种 RoPE 类型:
    - 'standard': ComplexRoPE (原始设计)
    - 'physio_v1': PhysioRoPE (分块分配 + 物理感知)
    - 'segmentation': SegmentationMRoPE (分割任务专用)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_parallel_scan: bool = False,  # 默认使用 sequential_scan 以节省显存
        # RoPE 配置
        rope_type: str = "segmentation",  # 'standard', 'physio_v1', 'segmentation'
        rope_base_temporal: float = 50000.0,
        rope_base_spatial: float = 10000.0,
        rope_temporal_ratio: float = 0.25,
        rope_num_axes: int = 2,  # 视频分割用2轴(t, spatial)
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_parallel_scan = use_parallel_scan
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.n_heads = self.d_inner // self.headdim
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # 1. Input Projection (Fusion of Multi-Scale if handled externally, 
        # but here we assume input is already fused or we project D_model -> D_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 2. Convolution (Optional in Mamba-3 but good for local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        # 3. SSM Projections
        # x -> dt, B, C
        # We need to project x to (dt_rank + 2 * d_state * n_heads)
        # B and C are rank-specific (d_state) per head
        self.x_proj = nn.Linear(
            self.d_inner, 
            self.dt_rank + self.n_heads * self.d_state * 2, 
            bias=False
        )
        
        # 4. Metric-Aware Parameters
        self.dt_params = MetricAwareTrapezoidalParameters(
            d_model=self.d_inner, # Uses projected x features effectively
            d_state=d_state,
            dt_rank=self.dt_rank,
            n_heads=self.n_heads,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
        )

        for param in self.dt_params.dt_proj.parameters():
            param.requires_grad = False
        
        # 5. RoPE (支持多种类型)
        self.rope_type = rope_type
        if rope_type == "standard":
            # 原始 ComplexRoPE
            self.rope = ComplexRoPE(dim=self.d_state)
        elif rope_type == "physio_v1":
            # PhysioRoPE v1 (分块分配)
            self.rope = PhysioRoPE(
                dim=self.d_state,
                num_axes=rope_num_axes,
                spacing_aware=True,
            )
        elif rope_type == "segmentation":
            # SegmentationMRoPE (我们的设计)
            self.rope = SegmentationMRoPE(
                dim=self.d_state,
                base_temporal=rope_base_temporal,
                base_spatial=rope_base_spatial,
                num_axes=rope_num_axes,
                temporal_dim_ratio=rope_temporal_ratio,
                learnable_base=True,
                spacing_aware=True,
            )
        else:
            raise ValueError(f"Unknown rope_type: {rope_type}") 

        # 6. Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # 7. Residual Connection Components (CRITICAL for preserving DINOv3 features!)
        self.norm = nn.LayerNorm(d_model)  # Pre-Norm
        self.out_norm = nn.LayerNorm(d_model)  # Output normalization to stabilize adapter output scale
        # Use learnable scale initialized to make adapter contribution ~10-20% of input
        # This is like LoRA's alpha parameter
        self.adapter_scale = nn.Parameter(torch.ones(1) * 0.5)  # Will be ~0.5 * normalized_output

    def _apply_rope(
        self, 
        x: torch.Tensor, 
        seq_len: int, 
        spacing: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        统一的RoPE应用接口，兼容不同类型的RoPE
        
        Args:
            x: [B, L, H, D] 输入张量
            seq_len: 序列长度
            spacing: 物理间距 (可选)
            positions: 显式位置编码 [B, L, num_axes] (可选)
            grid_shape: 网格形状 (可选)
        Returns:
            应用RoPE后的张量
        """
        if self.rope_type == "standard":
            # ComplexRoPE: 使用 seq_dim 参数
            return self.rope(x, seq_dim=1)
        elif self.rope_type == "physio_v1":
            # PhysioRoPE: 使用 positions 和 spacing
            return self.rope(x, positions=positions, spacing=spacing)
        elif self.rope_type == "segmentation":
            # SegmentationMRoPE: 使用 positions, spacing, grid_shape
            return self.rope(x, positions=positions, spacing=spacing, grid_shape=grid_shape)
        else:
            return x

    def forward(
        self, 
        x: torch.Tensor, 
        spacing: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ):
        """
        Args:
            x: [B, L, D_model]
            spacing: [B, L] or [B, 1] - Physical deltas for metric-aware modeling
            positions: [B, L, num_axes] - Explicit physical coordinates (optional)
            grid_shape: tuple - Implicit grid structure (optional)
        Returns:
            x + adapter_scale * adapter(norm(x))  # Residual connection!
        """
        batch, seq_len, _ = x.shape
        need_state = return_state or (state is not None)
        if state is not None:
            h_state, prev_bu_state, conv_cache = state
        else:
            h_state, prev_bu_state, conv_cache = None, None, None
        
        # 0. Pre-Norm (stabilizes training)
        x_normed = self.norm(x)
        
        # 1. Project
        xz = self.in_proj(x_normed) # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)
        
        # 2. Conv
        # Rearrange for conv: [B, D, L]
        new_conv_cache = None
        conv_cache_len = self.d_conv - 1
        if need_state and conv_cache_len > 0:
            if conv_cache is None:
                conv_cache = torch.zeros(batch, conv_cache_len, self.d_inner, device=x.device, dtype=x.dtype)
            x_in_cat = torch.cat([conv_cache, x_in], dim=1)  # [B, cache+L, D_inner]
            x_conv_cat = x_in_cat.transpose(1, 2)
            x_conv_cat = self.conv1d(x_conv_cat)[:, :, : x_in_cat.shape[1]]
            x_conv = x_conv_cat[:, :, conv_cache_len:]
            x_conv = self.act(x_conv).transpose(1, 2)  # [B, L, D_inner]
            new_conv_cache = x_in_cat[:, -conv_cache_len:, :]
        else:
            x_conv = x_in.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :seq_len]
            x_conv = self.act(x_conv).transpose(1, 2) # [B, L, D_inner]
        
        # 3. SSM Parameters Projection
        x_dbl = self.x_proj(x_conv) # [B, L, dt_rank + 2*n_heads*d_state]
        
        # Split parameters
        dt_rank_dim = self.dt_rank
        d_state_dim = self.n_heads * self.d_state
        
        dt_feat, B_flat, C_flat = torch.split(
            x_dbl, 
            [dt_rank_dim, d_state_dim, d_state_dim], 
            dim=-1
        )
        
        # 4. Metric-Aware dt calculation
        # Inject Spacing
        rope_spacing = spacing
        dt_spacing = None
        if spacing is not None:
            if spacing.dim() == 3:
                dt_spacing = spacing[..., 0]
            else:
                dt_spacing = spacing

        if dt_spacing is not None:
            if dt_spacing.dim() == 1:
                dt_spacing = dt_spacing.view(batch, 1, 1)
            elif dt_spacing.dim() == 2:
                dt_spacing = dt_spacing.unsqueeze(-1)
            
            # We use the spacing_proj from dt_params
            spacing_feat = self.dt_params.spacing_proj(dt_spacing)
            dt_inner = dt_feat + spacing_feat
        else:
            dt_inner = dt_feat
            
        dt_logits = self.dt_params.dt_head_proj(dt_inner)
        dt = F.softplus(dt_logits) # [B, L, n_heads]
        
        # Compute alpha, beta, gamma
        # A is [n_heads]
        A = -torch.exp(self.dt_params.A_log) 
        dt_A = dt * A.view(1, 1, -1) # [B, L, n_heads]
        alpha = torch.exp(dt_A)
        
        lam = torch.sigmoid(self.dt_params.lambda_param) if self.dt_params.learnable_lambda else self.dt_params.lambda_param
        lam = lam.view(1, 1, -1)
        
        beta = (1.0 - lam) * dt * alpha
        gamma = lam * dt
        
        # 5. Prepare B and C
        B_proj = B_flat.view(batch, seq_len, self.n_heads, self.d_state)
        C_proj = C_flat.view(batch, seq_len, self.n_heads, self.d_state)
        
        # Apply RoPE to B and C (兼容不同类型的RoPE)
        # 核心更新: 传递 positions 和 grid_shape 用于 Unified Metric Space-Time
        B_proj = self._apply_rope(B_proj, seq_len, rope_spacing, positions, grid_shape)
        C_proj = self._apply_rope(C_proj, seq_len, rope_spacing, positions, grid_shape)
        
        # 6. Scan (MIMO / SSD)
        # Reshape input u (x_conv) to [B, L, n_heads, headdim]
        u = x_conv.view(batch, seq_len, self.n_heads, self.headdim)
        
        # 选择扫描方法
        if self.use_parallel_scan and self.training and not need_state: 
            # 并行扫描 (较快但占用大量显存, 仅在训练且显存充足时建议开启)
            y = self.parallel_scan(u, alpha, beta, gamma, B_proj, C_proj)
        else:
            # 顺序扫描 (省内存, 适合大序列或推理)
            if need_state:
                initial_state = None
                if h_state is not None and prev_bu_state is not None:
                    initial_state = (h_state, prev_bu_state)
                y, (h_state, prev_bu_state) = self.sequential_scan(
                    u, alpha, beta, gamma, B_proj, C_proj, initial_state=initial_state, return_state=True
                )
            else:
                y = self.sequential_scan(u, alpha, beta, gamma, B_proj, C_proj)
        
        # 7. Output with Residual Connection (CRITICAL!)
        y = y.view(batch, seq_len, self.d_inner)
        y = y * F.silu(z) # Gating
        adapter_out = self.out_proj(y)
        
        # Normalize adapter output to match input scale (FIXES vanishing gradient!)
        adapter_out = self.out_norm(adapter_out)
        
        # Residual: preserve DINOv3 features, adapter learns refinement
        # x + scale * adapter(norm(x))
        out = x + self.adapter_scale * adapter_out
        if need_state:
            return out, (h_state, prev_bu_state, new_conv_cache)
        return out

    def parallel_scan(self, u, alpha, beta, gamma, B, C):
        """
        并行扫描算法 (Blelloch scan / Parallel prefix sum)
        比顺序扫描快 O(log n) 倍
        
        Args:
            u: [B, L, H, P] 输入序列
            alpha, beta, gamma: [B, L, H] 状态转移参数
            B, C: [B, L, H, N] SSM 参数
        
        Returns:
            y: [B, L, H, P] 输出序列
        """
        batch, seq_len, n_heads, headdim = u.shape
        d_state = B.shape[-1]
        
        # 简化版本: 将 trapezoidal 近似为标准形式 h_t = alpha_t * h_{t-1} + input_t
        # 其中 input_t = (beta_t + gamma_t) * B_t @ u_t
        
        # 1. 预计算所有 B @ u (外积)
        # B: [B, L, H, N], u: [B, L, H, P]
        # bu: [B, L, H, N, P]
        bu = torch.einsum('blhn,blhp->blhnp', B, u)
        
        # 2. 计算输入项 (简化 trapezoidal)
        input_term = (beta + gamma).unsqueeze(-1).unsqueeze(-1) * bu  # [B, L, H, N, P]
        
        # 3. 并行扫描核心
        # 对于 h_t = alpha_t * h_{t-1} + input_t
        # 使用结合律进行并行计算
        
        # alpha 累积: A_t = prod(alpha_0:t)
        log_alpha = torch.log(alpha.clamp(min=1e-8))  # [B, L, H]
        log_alpha_cumsum = torch.cumsum(log_alpha, dim=1)  # [B, L, H]
        A_cumsum = torch.exp(log_alpha_cumsum)  # [B, L, H]
        
        # 缩放输入项
        # h_t = A_t * (sum_{s=0}^{t} input_s / A_s)
        A_inv = 1.0 / A_cumsum.clamp(min=1e-8)
        scaled_input = input_term * A_inv.unsqueeze(-1).unsqueeze(-1)  # [B, L, H, N, P]
        
        # 累积和
        h_cumsum = torch.cumsum(scaled_input, dim=1)  # [B, L, H, N, P]
        
        # 恢复缩放
        h = h_cumsum * A_cumsum.unsqueeze(-1).unsqueeze(-1)  # [B, L, H, N, P]
        
        # 4. 输出: y_t = C_t @ h_t
        # C: [B, L, H, N], h: [B, L, H, N, P]
        y = torch.einsum('blhn,blhnp->blhp', C, h)  # [B, L, H, P]
        
        return y
    
    def sequential_scan(self, u, alpha, beta, gamma, B, C, initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_state: bool = False):
        """
        顺序扫描 (fallback, 较慢)
        Mamba-3 MIMO Scan (Trapezoidal)
        u: [B, L, H, P] (P=headdim)
        alpha, beta, gamma: [B, L, H]
        B, C: [B, L, H, N] (N=d_state)
        """
        batch, seq_len, n_heads, headdim = u.shape
        d_state = B.shape[-1]

        if initial_state is not None:
            h, prev_bu = initial_state
        else:
            h = torch.zeros(batch, n_heads, d_state, headdim, device=u.device, dtype=u.dtype)
            prev_bu = torch.zeros(batch, n_heads, d_state, headdim, device=u.device, dtype=u.dtype)

        ys = []
        
        for t in range(seq_len):
            # Get step params
            alpha_t = alpha[:, t, :, None, None] # [B, H, 1, 1]
            beta_t = beta[:, t, :, None, None]
            gamma_t = gamma[:, t, :, None, None]
            
            b_t = B[:, t, :, :, None] # [B, H, N, 1]
            u_t = u[:, t, :, None, :] # [B, H, 1, P]
            bu_t = b_t @ u_t          # [B, H, N, P] (Outer product)

            h = alpha_t * h + beta_t * prev_bu + gamma_t * bu_t
            prev_bu = bu_t
            
            # Output: y_t = C_t @ h_t
            c_t = C[:, t, :, None, :] # [B, H, 1, N]
            y_t = c_t @ h             # [B, H, 1, P]
            ys.append(y_t.squeeze(-2))
            
        y_stack = torch.stack(ys, dim=1) # [B, L, H, P]
        if return_state:
            return y_stack, (h, prev_bu)
        return y_stack
