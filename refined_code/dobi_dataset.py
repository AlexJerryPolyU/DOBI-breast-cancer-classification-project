"""
PyTorch Dataset class for fNIR data.

Handles data loading, channel conversion, and augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, labels_path=None, names_path=None, roi_path=None, channels=1):
        """
        初始化数据集

        参数：
        - data_path: 图像数据路径 (npy 文件)
        - labels_path: 标签数据路径 (npy 文件)；如果为 None，则不加载标签
        - names_path: 文件名数据路径 (npy 文件)
        - roi_path: ROI mask 数据路径 (可选，npy 文件)
        - channels: 输出通道数，1 表示单通道，3 表示三通道（默认 1）
        """
        # 加载数据
        self.data = np.load(data_path)  # 加载图像数据
        self.names = np.load(names_path)  # 加载文件名数据
        self.channels = channels

        # 如果提供了标签路径，则加载标签；否则设置为 None
        self.labels = None
        if labels_path is not None:
            self.labels = np.load(labels_path)  # 加载标签数据

        # 加载 ROI 数据（如果提供了路径）
        self.rois = None
        if roi_path is not None:
            self.rois = np.load(roi_path)  # 加载 ROI mask 数据

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """返回指定索引的数据、（可选）标签、文件名和 ROI"""
        image = torch.tensor(self.data[idx], dtype=torch.float32)  # 转为 tensor

        # 根据 channels 参数决定是否转换为三通道
        if self.channels == 3:
            image = image.repeat(1, 3, 1, 1)  # 单通道转 3 通道
        elif self.channels != 1:
            raise ValueError(f"Unsupported channel number: {self.channels}")

        name = self.names[idx]  # 获取文件名

        # 如果有 ROI，则返回 ROI
        if self.rois is not None:
            roi = torch.tensor(self.rois[idx], dtype=torch.float32)
            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return image, label, name, roi  # 返回 图像、标签、文件名、ROI
            else:
                return image, name, roi  # 无标签时，返回 图像、文件名、ROI

        # 没有 ROI 的情况
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label, name  # 返回 图像、标签、文件名
        else:
            return image, name  # 无标签时，返回 图像、文件名

