import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """图卷积网络层"""

    def __init__(self, d_in: int, d_out: int, device: str = "cuda"):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, device=device)
        self.linear2 = nn.Linear(d_in, d_out, device=device)
        self.norm = nn.BatchNorm1d(d_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        #Args:
            #x: (B, L, J, D)

        #Returns:
            #output: (B, L, J, D)


        # 调整邻接矩阵到相同设备
        adj = adj.to(x.device)

        # GCN计算
        out1 = self.linear1(x)  # (B*L, J, D_out)
        out1 = torch.bmm(out1.transpose(0, 1), adj.T).transpose(0, 1)  # 应用邻接矩阵

        out2 = self.linear2(x)  # (B*L, J, D_out)
        out2 = torch.bmm(out2, adj.unsqueeze(0).expand(x.shape, -1, -1))

        out = F.relu(out1 + out2)

        # 批规范化
        B_L, J, D = out.shape
        out = self.norm(out.reshape(B_L * J, D)).reshape(B_L, J, D)

        return out

class SpatialGCN(nn.Module):
    """空间GCN - 捕捉关键点间关系"""

    def __init__(self, d_in: int, d_out: int, num_joints: int = 17, device: str = "cuda"):
        super().__init__()
        self.num_joints = num_joints
        self.device = device

        self.gcn = GCNLayer(d_in, d_out, device=device)
        self.linear = nn.Linear(d_in, d_out, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(d_out, d_out * 2, device=device),
            nn.ReLU(),
            nn.Linear(d_out * 2, d_out, device=device)
        )
        self.norm = nn.LayerNorm(d_out)

        # 注册人体骨骼邻接矩阵
        self.register_buffer("adjacency", self._get_skeleton_adjacency(num_joints))

    def _get_skeleton_adjacency(self, num_joints: int) -> torch.Tensor:
        #\"\"\"获取人体骨骼拓扑邻接矩阵\"\"\"
        # Human3.6M骨骼连接 (17个关键点)
        edges = [
            (0, 1), (1, 2), (2, 3),  # 右臂
            (0, 4), (4, 5), (5, 6),  # 左臂
            (0, 7), (7, 8), (8, 9), (9, 10),  # 脊椎到右腿
            (8, 11), (11, 12), (12, 13),  # 左腿
            (1, 14), (14, 15),  # 右眼
            (1, 16), (16, 17),  # 左眼
        ]

        # 创建邻接矩阵
        adj = torch.zeros(num_joints, num_joints)
        for u, v in edges:
            if u < num_joints and v < num_joints:
                adj[u, v] = 1
                adj[v, u] = 1

        # 添加自环
        adj.fill_diagonal_(1)

        # 度数归一化
        D = torch.diag(adj.sum(dim=1) ** -0.5)
        adj_norm = D @ adj @ D

        return adj_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #Args:
        #x: (B, L, J, D)

    #Returns:
    #output: (B, L, J, D)

        B, L, J, D = x.shape

        # 展平用于GCN
        x_flat = x.reshape(B * L, J, D)

        # GCN处理
        x_gcn = self.gcn(x_flat, self.adjacency)

        # MLP处理
        x_out = self.mlp(self.norm(x_gcn))

        # 恢复形状
        output = x_out.reshape(B, L, J, -1)
        return output

class TemporalGCN(nn.Module):
    #"""时间GCN - 使用K-NN捕捉时间关系"""
    def __init__(self, d_in: int, d_out: int, k_neighbors: int = 2, device: str = "cuda"):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.device = device

        self.linear = nn.Linear(d_in, d_out, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(d_out, d_out * 2, device=device),
            nn.ReLU(),
            nn.Linear(d_out * 2, d_out, device=device)
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        #\"\"\"
        #Args:
        #x: (B, J, L, D)
        #causal: 是否应用因果mask

    #Returns:
    #output: (B, J, L, D)

#\"\"\"
        B, J, L, D = x.shape

        # 计算相似度矩阵
        x_flat = x.reshape(B * J, L, D)

        # 归一化特征
        x_norm = F.normalize(x_flat, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B*J, L, L)

        # 因果mask
        if causal:
            mask = torch.tril(torch.ones(L, L, device=x.device))
            sim = sim * mask.unsqueeze(0)

        # K-NN选择
        _, knn_indices = torch.topk(sim, min(self.k_neighbors, L), dim=-1)

        # 构造邻接矩阵
        adj = torch.zeros(B * J, L, L, device=x.device)
        for i in range(B * J):
            for j in range(L):
                adj[i, j, knn_indices[i, j]] = 1.0 / min(self.k_neighbors, L)

        # GCN聚合
        x_agg = torch.bmm(adj, x_flat)
        x_agg = self.linear(x_agg)
        x_agg = self.mlp(self.norm(x_agg))

        # 恢复形状
        output = x_agg.reshape(B, J, L, -1)
        return output

