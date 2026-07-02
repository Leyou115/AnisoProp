import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MetricAwareTrapezoidalParameters(nn.Module):
    """
    Computes the SSM parameters (alpha, beta, gamma) using the Metric-Aware Trapezoidal Rule.
    
    Physio-Mamba Innovation:
    - Injects physical 'spacing' into the calculation of dt (delta t).
    - Implements the Generalized Trapezoidal Discretization (Mamba-3).
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        dt_rank: int, 
        n_heads: int,
        dt_min: float = 0.001, 
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        learnable_lambda: bool = True,
        default_lambda: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.n_heads = n_heads
        
        # 1. Projections for dt (Information Flow Rate)
        # Project input x to low-rank dt_rank
        self.dt_proj = nn.Linear(d_model, dt_rank, bias=False)
        
        # Project spacing to dt_rank (Physio-Mamba Core)
        self.spacing_proj = nn.Linear(1, dt_rank, bias=False)
        
        # Project combined features to n_heads
        self.dt_head_proj = nn.Linear(dt_rank, n_heads, bias=True)
        
        # 2. Parameter A (Decay) - Optimized as in Mamba-2/3
        # A is usually -exp(A_log) to ensure stability
        self.A_log = nn.Parameter(torch.empty(n_heads))
        nn.init.normal_(self.A_log, mean=0.0, std=0.1) # Initialize A
        
        # 3. Parameter Lambda (Trapezoidal mixture coefficient)
        # lambda = 0.5 is standard trapezoidal. Mamba-3 allows it to be learnable.
        self.learnable_lambda = learnable_lambda
        if learnable_lambda:
            self.lambda_param = nn.Parameter(torch.full((n_heads,), default_lambda))
        else:
            self.register_buffer("lambda_param", torch.full((n_heads,), default_lambda))

        # dt initialization scheme
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        
        # Initialize dt bias
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_head_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_head_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias such that softplus(bias) is roughly exp(uniform(log(dt_min), log(dt_max)))
        dt = torch.exp(
            torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_head_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor, spacing: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, L, D]
            spacing: Physical spacing tensor [B, L] or [B, 1] or None. 
                     If None, assumes uniform spacing (1.0).
        Returns:
            alpha: [B, L, H] - Decay factor
            beta:  [B, L, H] - Input mixing factor (previous)
            gamma: [B, L, H] - Input mixing factor (current)
        """
        B, L, D = x.shape
        
        # 1. Compute Delta t (Step Size) with Spacing Injection
        x_proj = self.dt_proj(x) # [B, L, dt_rank]
        
        if spacing is not None:
            # Handle spacing shapes
            if spacing.dim() == 1:
                spacing = spacing.unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
            elif spacing.dim() == 2:
                spacing = spacing.unsqueeze(-1) # [B, L, 1]
            
            # Inject physics: spacing affects the projection
            # We add the spacing projection to the input projection before the final head projection
            spacing_feat = self.spacing_proj(spacing) # [B, L, dt_rank] or broadcastable
            dt_inner = x_proj + spacing_feat
        else:
            dt_inner = x_proj
            
        dt_logits = self.dt_head_proj(dt_inner) # [B, L, n_heads]
        dt = F.softplus(dt_logits) # [B, L, n_heads]
        
        # 2. Compute A (Decay Rate)
        # A must be negative for stability. 
        A = -torch.exp(self.A_log) # [H]
        
        # 3. Discretization (Trapezoidal Rule)
        # alpha = exp(dt * A)
        # dt * A broadcasts to [B, L, H]
        dt_A = dt * A.view(1, 1, -1)
        alpha = torch.exp(dt_A)
        
        # Lambda for trapezoidal mixing
        lam = torch.sigmoid(self.lambda_param) if self.learnable_lambda else self.lambda_param
        lam = lam.view(1, 1, -1)
        
        # beta = (1 - lambda) * dt * alpha
        # gamma = lambda * dt
        # Note: Mamba-3 formulation might vary slightly in notation, 
        # but the core idea is mixing current and previous inputs weighted by dt.
        
        # Using the formulation from our design doc:
        # h_t = alpha_t * h_{t-1} + beta_t * x_{t-1} + gamma_t * x_t
        
        beta = (1.0 - lam) * dt * alpha # [B, L, H]
        gamma = lam * dt                # [B, L, H]
        
        return alpha, beta, gamma


class ComplexRoPE(nn.Module):
    """
    Applies Rotary Positional Embeddings (RoPE) to simulate Complex-Valued State dynamics.
    
    Physio-Mamba Innovation:
    - Interprets 'Rotation' as 'Anatomical Continuity' tracking.
    - Applied to B and C matrices in the SSM.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies (theta)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for cos/sin
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        if (
            self._cos_cached is not None
            and self._cos_cached.size(0) >= seq_len
            and self._cos_cached.device == x.device
        ):
            return

        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [L, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [L, dim]
        
        self._cos_cached = emb.cos()[None, :, None, :] # [1, L, 1, D]
        self._sin_cached = emb.sin()[None, :, None, :] # [1, L, 1, D]

    def forward(self, x: torch.Tensor, seq_dim: int = 1):
        """
        Args:
            x: [B, L, N, D] or [B, L, D]
            seq_dim: Dimension corresponding to sequence length L
        """
        # Assuming x is [B, L, Heads, Head_Dim] or similar
        # We apply RoPE on the Head_Dim dimension
        
        seq_len = x.shape[seq_dim]
        self._update_cos_sin_tables(x, seq_len)
        
        # Slicing cached tables to current length
        cos = self._cos_cached[:, :seq_len, ...].to(x.dtype)
        sin = self._sin_cached[:, :seq_len, ...].to(x.dtype)
        
        return self._apply_rotary_pos_emb(x, cos, sin)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        # x: [B, L, H, D]
        # cos, sin: [1, L, 1, D]
        
        # Rotate every two elements: [-x2, x1]
        # We split D into D/2, D/2
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        
        # Standard RoPE rotation
        # x_rotated = x * cos + [-x2, x1] * sin
        
        # Reassemble [-x2, x1]
        neg_x2_x1 = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (neg_x2_x1 * sin)


class PhysioRoPE(nn.Module):
    """
    Physio-RoPE: Physics-Aware Rotary Position Embedding
    
    针对视频序列和 3D 医学影像的 RoPE 变体，融合以下创新：
    
    1. **Spacing-Aware Frequency Modulation (SAFM)**:
       - 将物理间距 (spacing) 融入旋转频率计算
       - θ_i = θ_base_i * spacing_factor
       - 理论依据：不同切片间距应产生不同的位置编码强度
    
    2. **Multi-Axis Decomposition (MAD)** (类似 Qwen3-VL M-RoPE):
       - 将维度分配给不同轴：时间/深度、高度、宽度
       - 每个轴有独立的频率基底
       - 适用于 3D 体数据和视频序列
    
    3. **Learnable Frequency Base (LFB)**:
       - 频率基底可学习，而非固定的 10000
       - 允许模型自适应最优的位置编码尺度
    
    References:
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
    - Qwen2-VL Technical Report (2024) - Multi-dimensional RoPE
    - Our Innovation: Spacing-aware frequency modulation for anisotropic medical data
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        num_axes: int = 1,  # 1=1D序列, 2=视频(T,HW), 3=3D体数据(D,H,W)
        learnable_base: bool = False,
        spacing_aware: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.num_axes = num_axes
        self.spacing_aware = spacing_aware
        
        # 维度分配给各个轴
        assert dim % 2 == 0, "dim must be even for RoPE"
        if num_axes == 1:
            self.dims_per_axis = [dim]
        elif num_axes == 2:
            # 视频: 时间轴 1/3, 空间轴 2/3 (参考 Qwen3-VL)
            t_dim = (dim // 3) // 2 * 2  # 确保偶数
            s_dim = dim - t_dim
            self.dims_per_axis = [t_dim, s_dim]
        elif num_axes == 3:
            # 3D: 深度 1/3, 高度 1/3, 宽度 1/3
            d_dim = (dim // 3) // 2 * 2
            h_dim = (dim // 3) // 2 * 2
            w_dim = dim - d_dim - h_dim
            self.dims_per_axis = [d_dim, h_dim, w_dim]
        else:
            raise ValueError(f"num_axes must be 1, 2, or 3, got {num_axes}")
        
        # 频率基底 (可学习或固定)
        if learnable_base:
            # 每个轴有独立的可学习基底
            self.base = nn.Parameter(torch.full((num_axes,), base))
        else:
            self.register_buffer("base", torch.full((num_axes,), base))
        
        # Spacing 调制网络 (如果启用)
        if spacing_aware:
            # 将 spacing 映射为频率调制因子
            self.spacing_mlp = nn.Sequential(
                nn.Linear(num_axes, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, num_axes),
                nn.Softplus()  # 确保输出为正
            )
            # 初始化为恒等映射附近
            nn.init.zeros_(self.spacing_mlp[0].weight)
            nn.init.ones_(self.spacing_mlp[0].bias)
            nn.init.zeros_(self.spacing_mlp[2].weight)
            nn.init.zeros_(self.spacing_mlp[2].bias)
        
        # 缓存
        self._cos_sin_cache = {}
    
    def _compute_inv_freq(self, axis_idx: int, spacing_factor: float = 1.0) -> torch.Tensor:
        """计算指定轴的逆频率，考虑 spacing 调制"""
        axis_dim = self.dims_per_axis[axis_idx]
        base = self.base[axis_idx] if self.base.dim() > 0 else self.base
        
        # 标准 RoPE 逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, axis_dim, 2, device=self.base.device).float() / axis_dim))
        
        # Spacing 调制：较大的间距 -> 较慢的旋转频率
        # 理论：物理距离大时，位置变化应更"缓慢"
        inv_freq = inv_freq * spacing_factor
        
        return inv_freq
    
    def _get_rotary_embedding(
        self, 
        positions: torch.Tensor, 
        axis_idx: int,
        spacing_factor: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为指定轴生成 cos/sin 嵌入
        
        Args:
            positions: [L] 位置索引
            axis_idx: 轴索引
            spacing_factor: 间距调制因子
        Returns:
            cos, sin: [L, axis_dim]
        """
        inv_freq = self._compute_inv_freq(axis_idx, spacing_factor)
        if device is not None:
            inv_freq = inv_freq.to(device)
        
        # positions: [L], inv_freq: [D/2]
        # freqs: [L, D/2]
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, D]
        
        cos = emb.cos()
        sin = emb.sin()
        
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        
        return cos, sin
    
    def forward(
        self, 
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        spacing: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        应用 Physio-RoPE
        
        Args:
            x: 输入张量 [B, L, H, D] 或 [B, L, D]
            positions: 位置张量，根据 num_axes:
                - 1D: [L] 或 None (自动生成 0,1,2,...)
                - 2D: [L, 2] 表示 (t, spatial_idx)
                - 3D: [L, 3] 表示 (d, h, w)
            spacing: 物理间距 [num_axes] 或 [B, num_axes]
                - 医学影像: [spacing_z, spacing_y, spacing_x]
                - 视频: [1/fps, 1.0] (时间间隔, 空间归一化)
            grid_shape: 网格形状，用于自动生成 positions
                - 2D: (T, H*W) 
                - 3D: (D, H, W)
        
        Returns:
            旋转后的张量，形状与输入相同
        """
        # 确定输入形状
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.unsqueeze(2)  # [B, L, 1, D]
            squeeze_back = True
        else:
            B, L, H, D = x.shape
            squeeze_back = False
        
        device = x.device
        dtype = x.dtype
        
        # 计算 spacing 调制因子
        if spacing is not None and self.spacing_aware:
            if spacing.dim() == 1:
                spacing = spacing.unsqueeze(0)  # [1, num_axes]
            spacing_factors = self.spacing_mlp(spacing)  # [B, num_axes] 或 [1, num_axes]
            spacing_factors = spacing_factors.mean(dim=0)  # [num_axes] 取平均
        else:
            spacing_factors = torch.ones(self.num_axes, device=device)
        
        # 生成位置索引
        if positions is None:
            if self.num_axes == 1:
                positions = torch.arange(L, device=device)
            elif grid_shape is not None:
                positions = self._generate_grid_positions(grid_shape, device)
            else:
                # 默认: 假设序列是展平的
                positions = torch.arange(L, device=device)
        
        # 计算各轴的 RoPE
        if self.num_axes == 1:
            cos, sin = self._get_rotary_embedding(
                positions, 0, spacing_factors[0].item(), device, dtype
            )
            cos = cos[None, :, None, :]  # [1, L, 1, D]
            sin = sin[None, :, None, :]
            
            x_rotated = self._apply_rotary(x, cos, sin)
        
        else:
            # 多轴情况：分别对不同维度切片应用不同轴的 RoPE
            x_parts = []
            dim_offset = 0
            
            for axis_idx in range(self.num_axes):
                axis_dim = self.dims_per_axis[axis_idx]
                x_axis = x[..., dim_offset:dim_offset + axis_dim]
                
                # 提取该轴的位置
                if positions.dim() == 1:
                    axis_positions = positions
                else:
                    axis_positions = positions[:, axis_idx]
                
                cos, sin = self._get_rotary_embedding(
                    axis_positions, axis_idx, spacing_factors[axis_idx].item(), device, dtype
                )
                cos = cos[None, :, None, :]
                sin = sin[None, :, None, :]
                
                x_axis_rotated = self._apply_rotary(x_axis, cos, sin)
                x_parts.append(x_axis_rotated)
                
                dim_offset += axis_dim
            
            x_rotated = torch.cat(x_parts, dim=-1)
        
        if squeeze_back:
            x_rotated = x_rotated.squeeze(2)
        
        return x_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """标准 RoPE 旋转操作"""
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        
        neg_x2_x1 = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (neg_x2_x1 * sin)
    
    def _generate_grid_positions(self, grid_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """生成网格位置索引"""
        if len(grid_shape) == 2:
            # 视频: (T, HW)
            T, HW = grid_shape
            t_pos = torch.arange(T, device=device).repeat_interleave(HW)
            s_pos = torch.arange(HW, device=device).repeat(T)
            return torch.stack([t_pos, s_pos], dim=-1)  # [T*HW, 2]
        
        elif len(grid_shape) == 3:
            # 3D: (D, H, W)
            D, H, W = grid_shape
            d_pos = torch.arange(D, device=device).repeat_interleave(H * W)
            h_pos = torch.arange(H, device=device).repeat_interleave(W).repeat(D)
            w_pos = torch.arange(W, device=device).repeat(D * H)
            return torch.stack([d_pos, h_pos, w_pos], dim=-1)  # [D*H*W, 3]
        
        else:
            raise ValueError(f"Unsupported grid_shape: {grid_shape}")


class SegmentationMRoPE(nn.Module):
    """
    Segmentation-MRoPE: 专为序列分割任务设计的多维旋转位置编码
    
    ═══════════════════════════════════════════════════════════════════
    与 VLM-MRoPE (Qwen3) 的本质区别：
    ═══════════════════════════════════════════════════════════════════
    
    VLM任务: 理解图像语义 → 生成文本
    我们的任务: 追踪目标 → 精确分割边界
    
    ┌─────────────────────────────────────────────────────────────────┐
    │  VLM: 关注"是什么" (What)                                        │
    │  我们: 关注"在哪里"+"怎么变" (Where + How it evolves)             │
    └─────────────────────────────────────────────────────────────────┘
    
    ═══════════════════════════════════════════════════════════════════
    核心创新设计：
    ═══════════════════════════════════════════════════════════════════
    
    1. **Temporal-Spatial Decoupled Frequencies (TSDF)**
       时空解耦频率设计
       
       - 时间轴(T): 使用**中低频**为主
         * 原因: 分割目标在相邻帧变化平滑，不需要对微小时间偏移过度敏感
         * 高频时间编码会导致相邻帧特征差异过大，破坏追踪连续性
         
       - 空间轴(H,W): 使用**全频谱**（高频+低频）
         * 原因: 边界需要高频捕捉精细变化，内部需要低频保持一致性
         
       实现: 时间轴频率基底更大（旋转更慢），空间轴频率基底更小（旋转更快）
       
       ```
       θ_time = 1 / (base_time^(2i/d))     # base_time = 50000 (慢旋转)
       θ_spatial = 1 / (base_spatial^(2i/d)) # base_spatial = 10000 (标准)
       ```
    
    2. **Anatomy-Continuity Encoding (ACE)**
       解剖连续性编码（医学影像特有）
       
       - 同一解剖结构在相邻切片应有"相似"的位置编码
       - 通过物理间距动态调整旋转步长：
         * 小间距(0.7mm): 相邻切片几乎相同 → 小角度旋转
         * 大间距(5mm): 相邻切片差异大 → 大角度旋转
         
       ```
       Δθ = θ_base * sigmoid(MLP(spacing))
       ```
    
    3. **Boundary-Interior Adaptive (BIA) - 可选**
       边界-内部自适应（分割特有）
       
       - 边界区域: 需要高频位置编码（对位置敏感）
       - 内部区域: 可用低频位置编码（位置容忍度高）
       - 通过注意力mask或显式区域标记实现
    
    4. **Cross-Frame Position Consistency (CFPC)**
       跨帧位置一致性
       
       - 问题: VLM的"每帧h,w从0开始"会破坏跨帧空间对应关系
       - 我们的设计: 空间位置在所有帧保持一致
         * 帧0的(h=5,w=10)和帧1的(h=5,w=10)应有相同的空间编码
         * 只有时间编码不同
       - 这样便于建立跨帧的空间对应关系（分割追踪的核心）
    
    ═══════════════════════════════════════════════════════════════════
    参数设计依据：
    ═══════════════════════════════════════════════════════════════════
    
    | 参数 | VLM典型值 | 我们的值 | 原因 |
    |------|----------|---------|------|
    | base_time | 10000 | 50000 | 时间变化要平滑 |
    | base_spatial | 10000 | 10000 | 空间需要精细 |
    | 时间维度占比 | 1/3 | 1/4 | 空间更重要 |
    | 空间坐标重置 | 每帧重置 | 全局一致 | 跨帧对应 |
    
    References:
    - Su et al., "RoFormer" (2021) - Original RoPE
    - 我们的创新: 针对分割任务的时空解耦设计
    """
    
    def __init__(
        self, 
        dim: int, 
        max_position_embeddings: int = 2048,
        base_temporal: float = 50000.0,   # 时间轴: 大基底 → 慢旋转 → 平滑追踪
        base_spatial: float = 10000.0,    # 空间轴: 标准基底 → 边界敏感
        num_axes: int = 3,                # 1=1D, 2=(t,s), 3=(t,h,w)
        temporal_dim_ratio: float = 0.25, # 时间维度占比 (VLM用1/3，我们用1/4)
        learnable_base: bool = True,      # 可学习基底
        spacing_aware: bool = True,       # 物理间距感知
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.num_axes = num_axes
        self.spacing_aware = spacing_aware
        
        assert dim % 2 == 0, "dim must be even for RoPE"
        
        # ═══════════════════════════════════════════════════════════════
        # 创新1: 时空解耦频率 (TSDF)
        # 时间轴用大基底(慢旋转)，空间轴用小基底(快旋转)
        # ═══════════════════════════════════════════════════════════════
        
        # 维度分配: 时间轴占 1/4，空间轴占 3/4
        if num_axes == 1:
            self.dim_temporal = dim
            self.dim_spatial = 0
        elif num_axes == 2:
            # (t, spatial)
            self.dim_temporal = int(dim * temporal_dim_ratio) // 2 * 2  # 确保偶数
            self.dim_spatial = dim - self.dim_temporal
        else:  # num_axes == 3
            # (t, h, w) - h和w共享空间维度
            self.dim_temporal = int(dim * temporal_dim_ratio) // 2 * 2
            self.dim_spatial = dim - self.dim_temporal
            # h和w各占空间维度的一半
            self.dim_h = self.dim_spatial // 2 // 2 * 2
            self.dim_w = self.dim_spatial - self.dim_h
        
        # 频率基底 (可学习 - 让模型自己学最优)
        # 确保是浮点数类型，否则nn.Parameter会报错
        if learnable_base:
            self.base_temporal = nn.Parameter(torch.tensor(float(base_temporal)))
            self.base_spatial = nn.Parameter(torch.tensor(float(base_spatial)))
        else:
            self.register_buffer("base_temporal", torch.tensor(float(base_temporal)))
            self.register_buffer("base_spatial", torch.tensor(float(base_spatial)))
        
        # ═══════════════════════════════════════════════════════════════
        # 创新2: 解剖连续性编码 (ACE)
        # 物理间距动态调整旋转步长
        # ═══════════════════════════════════════════════════════════════
        
        if spacing_aware:
            # 间距调制网络: spacing → 旋转步长缩放因子
            # 输入: [spacing_t, spacing_h, spacing_w] 或 [spacing_t, spacing_s]
            self.spacing_to_scale = nn.Sequential(
                nn.Linear(num_axes, 32),
                nn.GELU(),
                nn.Linear(32, num_axes),
            )
            # 初始化为恒等映射 (输出≈1)
            nn.init.zeros_(self.spacing_to_scale[0].weight)
            nn.init.zeros_(self.spacing_to_scale[0].bias)
            nn.init.zeros_(self.spacing_to_scale[2].weight)
            nn.init.zeros_(self.spacing_to_scale[2].bias)
            
            # 归一化参考值 (用于将物理间距归一化)
            # 视频: 默认30fps → spacing_t = 1/30
            # 医学: 典型层厚 2mm
            self.register_buffer("spacing_ref", torch.tensor([1/30.0, 1.0, 1.0][:num_axes]))
        
        # 缓存
        self._cache = {}
    
    def _compute_segmentation_rope(
        self,
        positions: torch.Tensor,      # [B, L, num_axes]
        spacing_scales: torch.Tensor, # [B, num_axes]
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算分割任务专用的 Interleaved M-RoPE
        
        核心设计:
        - 真正的频率交错 (True Frequency Interleaving): T -> H -> W -> T ...
        - 物理时间戳驱动 (Metric-Driven): positions 包含连续物理值
        """
        # 兼容旧接口 [L, ...] -> [1, L, ...]
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if spacing_scales.dim() == 1:
            spacing_scales = spacing_scales.unsqueeze(0)
            
        B, L, num_axes = positions.shape
        half_dim = self.dim // 2
        
        # 准备频率基底
        # bases: [num_axes]
        if num_axes == 1:
            bases = [self.base_temporal]
            dim_allocs = [self.dim_temporal]
        elif num_axes == 2:
            bases = [self.base_temporal, self.base_spatial]
            dim_allocs = [self.dim_temporal, self.dim_spatial]
        else: # 3
            bases = [self.base_temporal, self.base_spatial, self.base_spatial]
            dim_allocs = [self.dim_temporal, self.dim_h, self.dim_w]
            
        # 预计算每个轴的所有可能频率
        # axis_angles_all: list of [B, L, max_pairs]
        axis_angles_all = []
        
        for i in range(num_axes):
            # 该轴分配到的总维度对数 (budget)
            pairs_budget = dim_allocs[i] // 2
            if pairs_budget == 0:
                axis_angles_all.append(None)
                continue
                
            pos = positions[..., i].float() # [B, L]
            scale = spacing_scales[:, i]    # [B]
            base = bases[i]
            
            exponents = torch.arange(0, pairs_budget, device=device).float() * 2 / dim_allocs[i]
            inv_freq = 1.0 / (base ** exponents) # [pairs_budget]
            
            # 调制 scale: [B, 1] * [1, pairs] -> [B, pairs]
            inv_freq = inv_freq.unsqueeze(0) * scale.unsqueeze(1)
            
            # 计算角度: [B, L, 1] * [B, 1, pairs] -> [B, L, pairs]
            angles = torch.einsum("bl, bp -> blp", pos, inv_freq)
            axis_angles_all.append(angles)
            
        # ═══════════════════════════════════════════════════════════════
        # Interleaved Allocation (循环交错分配)
        # T, H, W, T, H, W ...
        # ═══════════════════════════════════════════════════════════════
        
        final_angles = torch.zeros(B, L, half_dim, device=device)
        
        # 追踪每个轴当前用到了第几个频率对
        axis_counters = [0] * num_axes
        
        # 循环填充 half_dim
        for i in range(half_dim):
            # 确定当前轮到哪个轴: i % num_axes
            # 但如果某个轴 budget 用完了，需要跳过
            # 我们用一个简单的 while 循环寻找下一个可用轴
            
            found_axis = -1
            # 尝试顺序: 从 (i % num_axes) 开始轮询
            start_axis_idx = i % num_axes
            
            for offset in range(num_axes):
                axis_idx = (start_axis_idx + offset) % num_axes
                
                if axis_idx < len(axis_angles_all) and axis_angles_all[axis_idx] is not None:
                    # 检查是否还有剩余配额
                    if axis_counters[axis_idx] < axis_angles_all[axis_idx].shape[-1]:
                        found_axis = axis_idx
                        break
            
            if found_axis != -1:
                # 取出频率值填入
                pair_idx = axis_counters[found_axis]
                final_angles[..., i] = axis_angles_all[found_axis][..., pair_idx]
                axis_counters[found_axis] += 1
            else:
                pass
                
        # 拼接 cos/sin
        emb = torch.cat([final_angles, final_angles], dim=-1) # [B, L, dim]
        
        return emb.cos().to(dtype), emb.sin().to(dtype)
    
    def forward(
        self, 
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        spacing: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        应用 Segmentation-MRoPE (Unified Metric Space-Time)
        
        核心机制:
        1. 物理时间戳 (Physical Timestamps):
           - 输入 spacing 被视为 delta_t (每步时差)
           - 累积生成连续时间坐标: t[i] = t[0] + sum(spacing[0...i])
           - 解决不规则采样(FPS变化/切片厚度不均)问题
           
        2. 批次独立位置 (Batch-dependent Positions):
           - 每个样本拥有独立的物理流形 (FPS不同 -> 时间流速不同)
           - Positions 形状: [B, L, num_axes]
           
        3. 交错频率 (Interleaved Frequencies):
           - 频率分配顺序: T -> H -> W -> T -> ...
           - 在特征维度上交织物理属性，而非简单的拼接
        
        Args:
            x: [B, L, H, D] 或 [B, L, D]
            positions: [B, L, num_axes] 或 [L, num_axes] (可选)
            spacing: [B, L] (delta_t) 或 [B, L, num_axes]
            grid_shape: 辅助生成 index positions (如果无 spacing)
        """
        # 处理输入形状
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.unsqueeze(2)
            squeeze_back = True
        else:
            B, L, H, D = x.shape
            squeeze_back = False
        
        device = x.device
        dtype = x.dtype
        
        # ═══════════════════════════════════════════════════════════════
        # 1. 构建物理坐标系 (Metric Coordinate System)
        # ═══════════════════════════════════════════════════════════════
        
        # 准备 spacing_summary 用于计算 scales
        # 我们需要一个 [B, num_axes] 的 tensor 来调节频率
        spacing_for_scale = None
        
        if positions is None:
            if spacing is not None:
                # -----------------------------------------------------------
                # Case A: 物理驱动 (Physics-Driven)
                # spacing 代表 delta (增量) -> 累积成 timestamps
                # -----------------------------------------------------------
                
                # 统一 spacing 形状到 [B, L, num_axes] 或 [B, L]
                if spacing.dim() == 1:
                    # [L] -> [1, L]
                    spacing = spacing.unsqueeze(0)
                if spacing.shape[0] != B:
                    # 如果 spacing 是 [1, L] 但 x 是 [B, L, ...], 广播
                    if spacing.shape[0] == 1:
                        spacing = spacing.expand(B, -1)
                    # 否则假设 spacing 已经匹配 B (注意: B可能是 batch*h*w)
                
                # 生成时间戳 t_pos
                if spacing.dim() == 2: # [B, L] -> 视为 delta_t
                    delta_t = spacing
                    # 累积求和得到时间戳
                    t_pos = torch.cumsum(delta_t, dim=1) # [B, L]
                    # 归零: 让第一个 token 时间为 0 (或者保留物理绝对位置? 通常相对位置更好)
                    t_pos = t_pos - t_pos[:, :1]
                    
                    # 构造 positions: [B, L, num_axes]
                    # 目前只知道 t，空间轴 (h, w) 默认为 0 (由上游控制是否传入)
                    # 如果需要空间感知，上游应直接传 positions 或 在这里注入 spatial_ids
                    zeros = torch.zeros_like(t_pos)
                    if self.num_axes == 2:
                        positions = torch.stack([t_pos, zeros], dim=-1)
                        # Scale summary: T=mean(delta), S=1.0
                        spacing_mean_t = delta_t.mean(dim=1, keepdim=True) # [B, 1]
                        spacing_for_scale = torch.cat([spacing_mean_t, torch.ones_like(spacing_mean_t)], dim=1)
                    else: # num_axes == 3
                        positions = torch.stack([t_pos, zeros, zeros], dim=-1)
                        # Scale summary: T=mean(delta), H=1.0, W=1.0
                        spacing_mean_t = delta_t.mean(dim=1, keepdim=True)
                        spacing_for_scale = torch.cat([spacing_mean_t, torch.ones_like(spacing_mean_t), torch.ones_like(spacing_mean_t)], dim=1)
                        
                elif spacing.dim() == 3: # [B, L, num_axes]
                    # 所有轴都有 delta
                    delta = spacing
                    pos_accum = torch.cumsum(delta, dim=1)
                    positions = pos_accum - pos_accum[:, :1, :]
                    spacing_for_scale = delta.mean(dim=1)
                    
            else:
                # -----------------------------------------------------------
                # Case B: 索引驱动 (Index-Driven / Fallback)
                # 没有 spacing，回退到标准 index positions
                # -----------------------------------------------------------
                if grid_shape is not None:
                    positions = self._generate_segmentation_positions(grid_shape, device)
                    # _generate 返回 [L, num_axes]，扩展到 [B, L, num_axes]
                    positions = positions.unsqueeze(0).expand(B, -1, -1)
                else:
                    # 纯 1D 序列
                    seq_idx = torch.arange(L, device=device).float()
                    positions = seq_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.num_axes)
                    if self.num_axes > 1:
                        # 如果是多轴但只有1D索引，把后面的轴置0
                        positions[..., 1:] = 0
                
                # 默认 scale 输入为 1
                spacing_for_scale = torch.ones(B, self.num_axes, device=device)

        else:
            # positions 已提供
            if positions.dim() == 2: # [L, axes]
                positions = positions.unsqueeze(0).expand(B, -1, -1)
            
            if spacing_for_scale is None:
                if spacing is not None:
                     # 尝试从 spacing 提取 scale
                     if spacing.dim() == 2:
                         mean_t = spacing.mean(dim=1, keepdim=True)
                         if mean_t.shape[0] == 1: mean_t = mean_t.expand(B, -1)
                         ones = torch.ones(B, self.num_axes - 1, device=device)
                         spacing_for_scale = torch.cat([mean_t, ones], dim=1)
                     elif spacing.dim() == 3:
                         spacing_for_scale = spacing.mean(dim=1)
                else:
                    spacing_for_scale = torch.ones(B, self.num_axes, device=device)

        # ═══════════════════════════════════════════════════════════════
        # 2. 解剖连续性编码 (ACE): 计算 Scales
        # ═══════════════════════════════════════════════════════════════
        if self.spacing_aware:
            # spacing_for_scale: [B, num_axes]
            spacing_normalized = spacing_for_scale / self.spacing_ref.to(device)
            
            # 学习缩放因子
            spacing_scales = torch.tanh(self.spacing_to_scale(spacing_normalized)) # [B, num_axes]
            spacing_scales = 1.0 + spacing_scales
        else:
            spacing_scales = torch.ones(B, self.num_axes, device=device)
        
        # ═══════════════════════════════════════════════════════════════
        # 3. 计算 Interleaved M-RoPE
        # ═══════════════════════════════════════════════════════════════
        cos, sin = self._compute_segmentation_rope(positions, spacing_scales, device, dtype)
        
        # 扩展维度以匹配广播: 
        # cos/sin: [B, L, D] -> [B, L, 1, D]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        
        # 应用旋转
        x_rotated = self._apply_rotary(x, cos, sin)
        
        if squeeze_back:
            x_rotated = x_rotated.squeeze(2)
        
        return x_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """标准 RoPE 旋转"""
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        neg_x2_x1 = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (neg_x2_x1 * sin)
    
    def _generate_segmentation_positions(
        self, 
        grid_shape: Tuple[int, ...], 
        device: torch.device,
    ) -> torch.Tensor:
        """
        生成分割任务专用的位置编码
        
        关键设计: 空间坐标在所有帧保持一致 (跨帧位置对应)
        
        ┌─────────────────────────────────────────────────────────────┐
        │ VLM: 每帧空间ID重置 → 破坏跨帧对应                           │
        │ 我们: 空间ID全局一致 → 帧0的(5,10)和帧1的(5,10)可直接对应    │
        └─────────────────────────────────────────────────────────────┘
        
        Args:
            grid_shape: (T, H, W) 或 (T, HW) 或 (D, H, W)
        """
        if len(grid_shape) == 2:
            # 视频 (T, HW) - 2轴模式
            T, HW = grid_shape
            
            # 时间位置: 0,0,0,...(HW个), 1,1,1,...(HW个), ...
            t_pos = torch.arange(T, device=device).repeat_interleave(HW)
            
            # 空间位置: 每帧相同 (跨帧对应!)
            # 0,1,2,...HW-1, 0,1,2,...HW-1, ...
            s_pos = torch.arange(HW, device=device).repeat(T)
            
            return torch.stack([t_pos, s_pos], dim=-1)
        
        elif len(grid_shape) == 3:
            # 3D (T/D, H, W) - 3轴模式
            T, H, W = grid_shape
            
            # 时间/深度位置
            t_pos = torch.arange(T, device=device).repeat_interleave(H * W)
            
            # 空间位置: 每个时间步/切片内相同 (跨帧/跨切片对应!)
            # h: 0,0,0...(W个),1,1,1...(W个),...,H-1,H-1..., 重复T次
            h_pos = torch.arange(H, device=device).repeat_interleave(W).repeat(T)
            # w: 0,1,2,...W-1, 重复T*H次
            w_pos = torch.arange(W, device=device).repeat(T * H)
            
            return torch.stack([t_pos, h_pos, w_pos], dim=-1)
        
        else:
            raise ValueError(f"Unsupported grid_shape: {grid_shape}")

