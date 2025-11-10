import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

from models.pose_magic import PoseMagic
from data.dataset import create_dataloaders
from utils.loss import PoseMagicLoss
from utils.metrics import MetricsTracker

def train_epoch(model: nn.Module,train_loader,optimizer,loss_fn,device,epoch: int,args):
#"""训练一个epoch"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        pose_2d = batch['pose_2d'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        # 前向传播
        pred_3d = model(pose_2d)

        # 计算损失
        losses = loss_fn(pred_3d, pose_3d)
        loss = losses['total']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            'loss': loss.item(),
            'pos': losses['pos'].item(),
            'vel': losses['vel'].item() if isinstance(losses['vel'], torch.Tensor) else losses['vel']
        })

    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(model: nn.Module,test_loader,device,):
    model.eval()
    metrics_tracker = MetricsTracker()
    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        pose_2d = batch['pose_2d'].to(device)
        pose_3d = batch['pose_3d'].to(device)

        # 前向传播
        pred_3d = model(pose_2d)

        # 计算指标
        metrics_tracker.update(pred_3d.cpu(), pose_3d.cpu())

    return metrics_tracker.get_mean()


def main():
    parser = argparse.ArgumentParser(description='训练Pose Magic模型')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='配置文件')
    parser.add_argument('--data-dir', type=str, help='数据目录')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='保存目录')
    parser.add_argument('--causal', action='store_true', help='使用因果版本')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='梯度裁剪值')
    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # 创建模型
    model = PoseMagic(causal=args.causal).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 损失函数
    loss_fn = PoseMagicLoss(lambda_vel=0.1).to(device)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    best_mpjpe = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"训练损失: {train_loss:.6f}")

        # 评估
        metrics = evaluate(model, test_loader, device)
        print(f"MPJPE: {metrics['mpjpe']:.2f}mm")
        print(f"MPJVE: {metrics['mpjve']:.2f}mm/s")
        print(f"Acc-Err: {metrics['acc_err']:.2f}mm/s²")

        # 保存最佳模型
        if metrics['mpjpe'] < best_mpjpe:
            best_mpjpe = metrics['mpjpe']
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, 'best_model.pth')
            )
            print(f"保存最佳模型，MPJPE: {best_mpjpe:.2f}mm")

        # 更新学习率
        scheduler.step()
if __name__ == '__main__':
    main()