import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class UsedCarDataset(Dataset):
    """
    自定义数据集类，用于将特征和标签转换为PyTorch张量。
    """

    def __init__(self, features, labels):
        # 确保输入是NumPy数组，并转换为float32张量
        self.features = torch.tensor(features, dtype=torch.float32)
        # 将标签转换为张量，并调整形状为 (N, 1)
        if isinstance(labels, pd.Series):
            labels = labels.values
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]