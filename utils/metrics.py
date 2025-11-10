import torch
import numpy as np


def mpjpe(pred_3d: torch.Tensor, gt_3d: torch.Tensor) -> float:
    """Mean Per Joint Position Error (MPJPE)"""
    errors = torch.norm(pred_3d - gt_3d, dim=-1)
    return errors.mean().item() * 1000  # 转换为mm


def mpjve(pred_3d: torch.Tensor, gt_3d: torch.Tensor) -> float:
    """Mean Per Joint Velocity Error (MPJVE)"""
    pred_vel = pred_3d[:, 1:] - pred_3d[:, :-1]
    gt_vel = gt_3d[:, 1:] - gt_3d[:, :-1]

    errors = torch.norm(pred_vel - gt_vel, dim=-1)
    return errors.mean().item() * 1000


def acceleration_error(pred_3d: torch.Tensor, gt_3d: torch.Tensor) -> float:
    """Acceleration Error"""
    pred_vel = pred_3d[:, 1:] - pred_3d[:, :-1]
    gt_vel = gt_3d[:, 1:] - gt_3d[:, :-1]

    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]

    errors = torch.norm(pred_acc - gt_acc, dim=-1)
    return errors.mean().item() * 1000


class MetricsTracker:
    """指标追踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mpjpe_list = []
        self.mpjve_list = []
        self.acc_err_list = []

    def update(self, pred_3d: torch.Tensor, gt_3d: torch.Tensor):
        self.mpjpe_list.append(mpjpe(pred_3d, gt_3d))
        self.mpjve_list.append(mpjve(pred_3d, gt_3d))
        self.acc_err_list.append(acceleration_error(pred_3d, gt_3d))

    def get_mean(self) -> dict:
        return {
            'mpjpe': np.mean(self.mpjpe_list),
            'mpjve': np.mean(self.mpjve_list),
            'acc_err': np.mean(self.acc_err_list),
        }
