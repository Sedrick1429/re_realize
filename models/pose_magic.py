import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mamba import BidirectionalMamba, UnidirectionalMamba
from .gcn import SpatialGCN, TemporalGCN

class MagicBlock(nn.Module):
    #"""双流Magic块 - 融合Mamba和GCN"""
    def __init__(
            self,
            d_model: int = 128,
            num_joints: int = 17,
            k_neighbors: int = 2,
            causal: bool = False,
            device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints
        self.causal = causal
        self.device = device

        # Mamba流
        if causal:
            self.spatial_mamba = UnidirectionalMamba(d_model, device=device)
            self.temporal_mamba = UnidirectionalMamba(d_model, device=device)
        else:
            self.spatial_mamba = BidirectionalMamba(d_model, device=device)
            self.temporal_mamba = BidirectionalMamba(d_model, device=device)

        # GCN流
        self.spatial_gcn = SpatialGCN(d_model, d_model, num_joints, device=device)
        self.temporal_gcn = TemporalGCN(d_model, d_model, k_neighbors, device=device)

        # 自适应融合
        self.fusion_proj = nn.Linear(d_model * 2, 2, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #\"\"\"
        #Args:
        #x: (B, L, J, D)

    #Returns:
    #output: (B, L, J, D)

#\"\"\"
        B, L, J, D = x.shape

        # ===== 空间处理 =====
        # Mamba处理
        x_spatial = x.reshape(B * L, J, D)
        x_spatial_mamba = self.spatial_mamba(x_spatial)
        x_spatial_mamba = x_spatial_mamba.reshape(B, L, J, D)

        # GCN处理
        x_spatial_gcn = self.spatial_gcn(x)

        # 自适应融合
        x_spatial_fused = self._adaptive_fusion(x_spatial_mamba, x_spatial_gcn)

        # ===== 时间处理 =====
        x_temporal = x.permute(0, 2, 1, 3)  # (B, J, L, D)

        # Mamba处理
        x_temporal_mamba = self.temporal_mamba(x_temporal.reshape(B * J, L, D))
        x_temporal_mamba = x_temporal_mamba.reshape(B, J, L, D)

        # GCN处理
        x_temporal_gcn = self.temporal_gcn(x_temporal, causal=self.causal)

        # 自适应融合
        x_temporal_fused = self._adaptive_fusion(x_temporal_mamba, x_temporal_gcn)

        # 恢复为 (B, L, J, D)
        x_temporal_fused = x_temporal_fused.permute(0, 2, 1, 3)

        # 最终融合
        output = x_spatial_fused + x_temporal_fused

        return output

    def _adaptive_fusion(self, x_mamba: torch.Tensor, x_gcn: torch.Tensor) -> torch.Tensor:
        #\"\"\"自适应融合两个流的输出\"\"\"
        # 连接特征
        x_concat = torch.cat([x_mamba, x_gcn], dim=-1)

        # 计算融合权重
        if len(x_concat.shape) == 4:  # (B, L, J, 2D)
            B, L, J, _ = x_concat.shape
            x_concat_flat = x_concat.reshape(B * L * J, -1)
            alpha = torch.softmax(self.fusion_proj(x_concat_flat), dim=-1)
            alpha = alpha.reshape(B, L, J, 2)
        else:  # (B, L, 2D)
            alpha = torch.softmax(self.fusion_proj(x_concat), dim=-1)

        # 加权融合
        output = alpha[..., 0:1] * x_mamba + alpha[..., 1:2] * x_gcn
        return output

class PoseMagic(nn.Module):
    #"""Pose Magic: 高效3D人体姿态估计"""
    def __init__(
            self,
            num_layers: int = 26,
            d_model: int = 128,
            d_prime: int = 512,
            num_joints: int = 17,
            k_neighbors: int = 2,
            causal: bool = False,
            device: str = "cuda",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_joints = num_joints
        self.device = device

        # 输入投影
        self.input_proj = nn.Linear(3, d_model, device=device)

        # 位置编码
        self.register_buffer(
            "pos_embed",
            self._get_positional_embedding(256, num_joints, d_model)
        )

        # Magic blocks
        self.magic_blocks = nn.ModuleList([
            MagicBlock(d_model, num_joints, k_neighbors, causal, device=device)
            for _ in range(num_layers)
        ])

        # 特征投影
        self.feature_proj = nn.Linear(d_model, d_prime, device=device)
        self.feature_act = nn.Tanh()

        # 输出头
        self.output_head = nn.Linear(d_prime, 3, device=device)

    def _get_positional_embedding(
            self,
            max_len: int,
            num_joints: int,
            d_model: int
    ) -> torch.Tensor:
        #\"\"\"计算位置编码\"\"\"
        pe = torch.zeros(max_len, num_joints, d_model)

        # 关键点位置编码
        pos = torch.arange(num_joints).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, :, 0::2] = torch.sin(pos * div_term).unsqueeze(0)
        if d_model % 2 == 1:
            pe[:, :, 1::2] = torch.cos(pos * div_term).unsqueeze(0)[:, :, :-1]
        else:
            pe[:, :, 1::2] = torch.cos(pos * div_term).unsqueeze(0)

        # 时间位置编码
        pos_t = torch.arange(max_len).unsqueeze(1)
        pe_t = torch.zeros(max_len, 1, d_model)
        pe_t[:, :, 0::2] = torch.sin(pos_t * div_term).unsqueeze(1)
        if d_model % 2 == 1:
            pe_t[:, :, 1::2] = torch.cos(pos_t * div_term).unsqueeze(1)[:, :, :-1]
        else:
            pe_t[:, :, 1::2] = torch.cos(pos_t * div_term).unsqueeze(1)

        # 合并
        pe = pe + pe_t
        return pe

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:

        #Args:
        #x_2d: (B, T, J, 3) - 2
        #D关键点序列

        #Returns:
        #x_3d: (B, T, J, 3) - 3
        #D姿态预测

        B, T, J, _ = x_2d.shape

        # 输入投影
        x = self.input_proj(x_2d)

        # 添加位置编码
        x = x + self.pos_embed[:T, :J, :].unsqueeze(0).to(x.device)

        # 通过Magic blocks
        for block in self.magic_blocks:
            x = block(x)

        # 特征投影
        x = self.feature_proj(x)
        x = self.feature_act(x)

        # 输出
        x_3d = self.output_head(x)

        return x_3d

