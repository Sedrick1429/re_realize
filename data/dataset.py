import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class Human36MDataset(Dataset):
    """Human3.6M 数据集加载器"""

    def __init__(
            self,
            data_dir: str,
            split: str = 'train',
            seq_len: int = 243,
            subjects: list = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len

        # 默认训练/测试集划分
        if subjects is None:
            if split == 'train':
                self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
            else:  # test
                self.subjects = ['S9', 'S11']
        else:
            self.subjects = subjects

        self.data = self._load_data()

    def _load_data(self):
        """加载Human3.6M数据"""
        data_2d_list = []
        data_3d_list = []

        for subject in self.subjects:
            # 从预处理文件加载
            npz_file_2d = os.path.join(
                self.data_dir,
                f'data_2d_h36m_gt_{subject}.npz'
            )
            npz_file_3d = os.path.join(
                self.data_dir,
                f'data_3d_h36m_{subject}.npz'
            )

            if os.path.exists(npz_file_2d) and os.path.exists(npz_file_3d):
                data_2d = np.load(npz_file_2d)['positions_2d']  # (A, T, J, 2)
                data_3d = np.load(npz_file_3d)['positions_3d']  # (A, T, J, 3)

                data_2d_list.append(data_2d)
                data_3d_list.append(data_3d)

        return {
            'data_2d': np.concatenate(data_2d_list, axis=0),
            'data_3d': np.concatenate(data_3d_list, axis=0),
        }

    def __len__(self):
        num_sequences = self.data['data_2d'].shape * self.data['data_2d'].shape
        return max(1, num_sequences // self.seq_len)

    def __getitem__(self, idx):
        """获取序列样本"""
        data_2d_all = self.data['data_2d']
        data_3d_all = self.data['data_3d']

        # 计算起始位置
        total_frames = data_2d_all.shape * data_2d_all.shape
        start_frame = (idx * self.seq_len) % (total_frames - self.seq_len)

        # 获取序列
        seq_2d = []
        seq_3d = []

        for i in range(self.seq_len):
            frame_idx = (start_frame + i) % total_frames
            action_idx = frame_idx // data_2d_all.shape
            frame_in_action = frame_idx % data_2d_all.shape

            seq_2d.append(data_2d_all[action_idx, frame_in_action])
            seq_3d.append(data_3d_all[action_idx, frame_in_action])

        seq_2d = np.stack(seq_2d)  # (T, J, 2)
        seq_3d = np.stack(seq_3d)  # (T, J, 3)

        # 添加置信度
        seq_2d = np.concatenate([seq_2d, np.ones((seq_2d.shape, seq_2d.shape, 1))], axis=-1)

        return {
            'pose_2d': torch.from_numpy(seq_2d).float(),
            'pose_3d': torch.from_numpy(seq_3d).float(),
        }


def create_dataloaders(
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        seq_len: int = 243,
):
    """创建训练和验证数据加载器"""

    train_dataset = Human36MDataset(
        data_dir=data_dir,
        split='train',
        seq_len=seq_len,
    )

    test_dataset = Human36MDataset(
        data_dir=data_dir,
        split='test',
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
