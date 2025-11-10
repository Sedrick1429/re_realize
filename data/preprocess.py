import os
import numpy as np
import h5py
import argparse
from pathlib import Path


class H36MPreprocessor:
    """Human3.6M 数据预处理器"""

    def __init__(self, raw_dir, output_dir):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_2d_poses(self, h5_file):
        """从HDF5文件中提取2D姿态"""
        with h5py.File(h5_file, 'r') as f:
            # Human3.6M 2D姿态格式: (actions, frames, joints, 2)
            positions_2d = f['positions_2d'][:]
        return positions_2d

    def extract_3d_poses(self, h5_file):
        """从HDF5文件中提取3D姿态"""
        with h5py.File(h5_file, 'r') as f:
            # Human3.6M 3D姿态格式: (actions, frames, joints, 3)
            positions_3d = f['positions_3d'][:]
        return positions_3d

    def normalize_2d(self, poses_2d):
        """归一化2D姿态"""
        # 沿时间和关键点维度计算均值和标准差
        mean = poses_2d.reshape(-1, poses_2d.shape[-1]).mean(axis=0)
        std = poses_2d.reshape(-1, poses_2d.shape[-1]).std(axis=0)
        std[std == 0] = 1  # 避免除以0

        normalized = (poses_2d - mean) / std
        return normalized, mean, std

    def normalize_3d(self, poses_3d):
        """归一化3D姿态"""
        # 移除根节点偏差
        root = poses_3d[..., 0:1, :]  # (A, T, 1, 3)
        poses_3d_centered = poses_3d - root

        # 归一化
        mean = poses_3d_centered.reshape(-1, 3).mean(axis=0)
        std = poses_3d_centered.reshape(-1, 3).std(axis=0)
        std[std == 0] = 1

        normalized = (poses_3d_centered - mean) / std
        return normalized, mean, std

    def process_subject(self, subject_id):
        """处理单个被试的数据"""
        print(f"处理被试: {subject_id}")

        # 2D姿态文件
        file_2d = os.path.join(self.raw_dir, f'data_2d_h36m_gt_{subject_id}.npz')
        # 3D姿态文件
        file_3d = os.path.join(self.raw_dir, f'data_3d_h36m_{subject_id}.npz')

        if not os.path.exists(file_2d) or not os.path.exists(file_3d):
            print(f"⚠ 文件不存在: {subject_id}")
            return False

        # 加载数据
        data_2d = np.load(file_2d, allow_pickle=True)['positions_2d'].astype(np.float32)
        data_3d = np.load(file_3d, allow_pickle=True)['positions_3d'].astype(np.float32)

        print(f"  2D姿态形状: {data_2d.shape}")
        print(f"  3D姿态形状: {data_3d.shape}")

        # 归一化
        data_2d_norm, mean_2d, std_2d = self.normalize_2d(data_2d)
        data_3d_norm, mean_3d, std_3d = self.normalize_3d(data_3d)

        # 保存处理后的数据
        output_2d = os.path.join(self.output_dir, f'data_2d_h36m_gt_{subject_id}.npz')
        output_3d = os.path.join(self.output_dir, f'data_3d_h36m_{subject_id}.npz')

        np.savez_compressed(output_2d, positions_2d=data_2d_norm,
                            mean=mean_2d, std=std_2d)
        np.savez_compressed(output_3d, positions_3d=data_3d_norm,
                            mean=mean_3d, std=std_3d)

        print(f"  ✓ 数据已保存")
        return True

    def process_all(self, subjects=None):
        """处理所有被试数据"""
        if subjects is None:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        print("=" * 60)
        print("Human3.6M 数据预处理")
        print("=" * 60)
        print(f"原始数据目录: {self.raw_dir}")
        print(f"输出目录: {self.output_dir}")
        print()

        success_count = 0
        for subject in subjects:
            if self.process_subject(subject):
                success_count += 1

        print()
        print("=" * 60)
        print(f"✓ 处理完成: {success_count}/{len(subjects)} 个被试")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Human3.6M 数据预处理')
    parser.add_argument('--raw-dir', type=str, required=True,
                        help='原始数据目录')
    parser.add_argument('--output-dir', type=str, default='./data/processed',
                        help='输出目录')
    parser.add_argument('--subjects', type=str, default='S1,S5,S6,S7,S8,S9,S11',
                        help='被试列表，逗号分隔')

    args = parser.parse_args()

    subjects = args.subjects.split(',')
    preprocessor = H36MPreprocessor(args.raw_dir, args.output_dir)
    preprocessor.process_all(subjects)


if __name__ == '__main__':
    main()
