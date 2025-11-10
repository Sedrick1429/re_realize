import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseMagicLoss(nn.Module):
    #"""Pose Magic损失函数"""
    def __init__(self, lambda_vel: float = 0.1, lambda_acc: float = 0.01):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(
            self,
            pred_3d: torch.Tensor,
            gt_3d: torch.Tensor,
    ) -> dict:

        #Args:
        #pred_3d: (B, T, J, 3)
        #预测3D姿态
        #gt_3d: (B, T, J, 3)
        #真实3D姿态

        #Returns:
        #损失字典
        # 确保张量在同一设备
        gt_3d = gt_3d.to(pred_3d.device)

        # 位置损失
        loss_3d = self.l1_loss(pred_3d, gt_3d)

        # 速度损失
        if gt_3d.shape > 1:
            pred_vel = pred_3d[:, 1:] - pred_3d[:, :-1]
            gt_vel = gt_3d[:, 1:] - gt_3d[:, :-1]
            loss_vel = self.l1_loss(pred_vel, gt_vel)
        else:
            loss_vel = 0.0

        # 加速度损失
        if gt_3d.shape > 2:
            pred_acc = pred_3d[:, 2:] - 2 * pred_3d[:, 1:-1] + pred_3d[:, :-2]
            gt_acc = gt_3d[:, 2:] - 2 * gt_3d[:, 1:-1] + gt_3d[:, :-2]
            loss_acc = self.mse_loss(pred_acc, gt_acc)
        else:
            loss_acc = 0.0

        # 总损失
        total_loss = loss_3d + self.lambda_vel * loss_vel + self.lambda_acc * loss_acc

        return {
            'total': total_loss,
            'pos': loss_3d,
            'vel': loss_vel if isinstance(loss_vel, torch.Tensor) else torch.tensor(loss_vel, device=pred_3d.device),
            'acc': loss_acc if isinstance(loss_acc, torch.Tensor) else torch.tensor(loss_acc, device=pred_3d.device),
        }


