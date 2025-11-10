import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MambaBlock(nn.Module):
    """Mamba State Space Model Block"""

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
            dt_rank: Optional[int] = None,
            dt_scale: float = 1.0,
            dt_init: str = "random",
            dt_init_floor: float = 1e-4,
            device: str = "cuda",
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank or int(np.ceil(d_model / 16))
        self.device = device

        # 输入投影
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            device=device,
        )

        # 激活函数
        self.act = nn.SiLU()

        # SSM参数
        self.A_log = nn.Parameter(
            torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
            .log()
            .repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt投影
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, device=device)

        # 初始化dt_proj
        dt_init_std = self.dt_rank ** -0.5
        with torch.no_grad():
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            self.dt_proj.bias.fill_(np.log(dt_init_floor))

        # B和C投影
        self.B_proj = nn.Linear(d_model, d_state, bias=False, device=device)
        self.C_proj = nn.Linear(d_model, d_state, bias=False, device=device)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, L, D) 张量

        Returns:
            output: (B, L, D) 张量
        """
        batch, length, dim = x.shape

        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x_val, z = xz.chunk(2, dim=-1)  # 各为 (B, L, D_inner)

        # 转置用于卷积: (B, L, D) -> (B, D, L)
        x_val = x_val.transpose(1, 2)

        # 1D卷积
        x_val = self.conv1d(x_val)[:, :, :length]  # (B, D_inner, L)

        # 转置回: (B, D, L) -> (B, L, D)
        x_val = x_val.transpose(1, 2)

        # 激活
        x_val = self.act(x_val)  # (B, L, D_inner)

        # 获取A矩阵（离散化）
        A = -torch.exp(self.A_log.float())  # (D_inner, d_state)

        # dt离散化
        dt = self.dt_proj(x[:, :, :self.dt_rank])  # (B, L, D_inner)
        dt = F.softplus(dt)  # (B, L, D_inner)

        # B和C
        B = self.B_proj(x)  # (B, L, d_state)
        C = self.C_proj(x)  # (B, L, d_state)

        # SSM计算
        y = self._ssm_forward(x_val, A, B, C, dt)  # (B, L, D_inner)

        # 门控单元
        y = y * self.act(z)

        # 输出投影
        output = self.out_proj(y)  # (B, L, D)

        return output

    def _ssm_forward(
            self,
            x: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            C: torch.Tensor,
            dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        SSM前向传播（递归方式）

        Args:
            x: (B, L, D_inner)
            A: (D_inner, d_state)
            B: (B, L, d_state)
            C: (B, L, d_state)
            dt: (B, L, D_inner)

        Returns:
            y: (B, L, D_inner)
        """
        batch, length, d_inner = x.shape
        d_state = B.shape[-1]
        device = x.device

        # 初始化隐藏状态
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=x.dtype)
        ys = []

        # 递归计算
        for t in range(length):
            # 离散A
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # (B, D_inner, d_state)

            # 更新隐藏状态
            h = dA * h + (dt[:, t, :].unsqueeze(-1) * x[:, t, :].unsqueeze(-1)) * B[:, t, :].unsqueeze(1)

            # 输出
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D_inner)

class BidirectionalMamba(nn.Module):
    """双向Mamba块 - 用于离线推理"""
    def __init__(self, d_model: int, device: str = "cuda", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 三个Mamba流
        self.mamba_f = MambaBlock(d_model, device=device, **kwargs)  # 前向
        self.mamba_b = MambaBlock(d_model, device=device, **kwargs)  # 后向
        self.mamba_i = MambaBlock(d_model, device=device, **kwargs)  # 独立

        # 归一化
        self.norm = nn.LayerNorm(d_model)

        # 投影矩阵
        self.Wp1 = nn.Linear(d_model, d_model)
        self.Wp2 = nn.Linear(d_model, d_model)
        self.Wp3 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)

        Returns:
            output: (B, L, D)
        """
        # 归一化
        x_norm = self.norm(x)

        # 前向流
        x_f = F.gelu(torch.matmul(x_norm, self.Wp1.weight.T))
        x_f = self.mamba_f(x_f)

        # 后向流（翻转序列）
        x_b = torch.flip(x_norm, dims=[3,3,3])
        x_b = F.gelu(torch.matmul(x_b, self.Wp1.weight.T))
        x_b = self.mamba_b(x_b)
        x_b = torch.flip(x_b, dims=[3,3,3])

        # 独立流
        x_i = F.gelu(torch.matmul(x_norm, self.Wp2.weight.T))

        # 乘法门控
        xa = x_f * x_i + x_b * x_i

        # 跳过连接和投影
        output = x + torch.matmul(xa, self.Wp3.weight.T)

        return output
class UnidirectionalMamba(nn.Module):
    #"""单向（因果）Mamba块 - 用于实时推理"""
    def __init__(self, d_model: int, device: str = "cuda", **kwargs):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # 两个Mamba流（仅前向和独立）
        self.mamba_f = MambaBlock(d_model, device=device, **kwargs)
        self.mamba_i = MambaBlock(d_model, device=device, **kwargs)

        # 归一化
        self.norm = nn.LayerNorm(d_model)

        # 投影矩阵
        self.Wp1 = nn.Linear(d_model, d_model)
        self.Wp2 = nn.Linear(d_model, d_model)
        self.Wp3 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #Args:
        #x: (B, L, D)

    #Returns:
    #output: (B, L, D)

        # 归一化
        x_norm = self.norm(x)

        # 前向流（因果）
        x_f = F.gelu(torch.matmul(x_norm, self.Wp1.weight.T))
        x_f = self.mamba_f(x_f)

        # 独立流
        x_i = F.gelu(torch.matmul(x_norm, self.Wp2.weight.T))

        # 乘法门控（仅前向+独立）
        xa = x_f * x_i

        # 跳过连接
        output = x + torch.matmul(xa, self.Wp3.weight.T)

        return output


