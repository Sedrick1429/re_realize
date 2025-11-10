import torch
import argparse
from models.pose_magic import PoseMagic
from data.dataset import create_dataloaders
from utils.metrics import MetricsTracker


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseMagic().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, test_loader = create_dataloaders(args.data_dir, batch_size=8)
    metrics = MetricsTracker()

    for batch in test_loader:
        pose_2d, pose_3d = batch['pose_2d'].to(device), batch['pose_3d'].to(device)
        pred = model(pose_2d)
        metrics.update(pred.cpu(), pose_3d.cpu())

    result = metrics.get_mean()
    print(f"MPJPE: {result['mpjpe']:.2f}mm")
    print(f"MPJVE: {result['mpjve']:.2f}mm/s")


if __name__ == '__main__':
    evaluate()
