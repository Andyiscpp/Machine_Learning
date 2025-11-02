import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score  # 计算均方误差mse和R^2

from Deeptrain3.UsedCarDataset import UsedCarDataset


class ModelTrainer:
    """
    模型训练器，负责模型的初始化、训练循环、早停和评估。
    """

    def __init__(self, model, random_state=42):
        self.model = model  # 接收一个已经实例化的 PyTorch 模型作为参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 检查是否有可用的 GPU (cuda)。如果有，self.device 设置为 cuda，否则使用 cpu
        self.model.to(self.device)  # 将模型的所有参数和缓冲区移动到指定的设备（GPU 或 CPU）

        # 训练记录
        self.train_losses = []
        self.val_losses = []  # 存储每个 epoch 的训练和验证损失
        self.train_r2_scores = []
        self.val_r2_scores = []
        self.best_val_loss = float('inf')
        self.best_model_state = None  # 存储验证损失最低时的模型权重（“状态字典”）

        torch.manual_seed(random_state)
        print(f"训练使用设备: {self.device}")

    def evaluate(self, X, y):
        """
        评估模型性能并返回R²分数。
        """
        self.model.eval()  # 将模型切换到“评估模式”

        with torch.no_grad():  # 在这个代码块中不要计算梯度
            features = torch.tensor(X, dtype=torch.float32).to(self.device)
            # (修复) 确保 y 是 numpy 数组
            if isinstance(y, pd.Series):
                y_values = y.values
            else:
                y_values = y
            labels = torch.tensor(y_values, dtype=torch.float32).to(self.device).view(-1, 1)

            outputs = self.model(features)

            y_pred = outputs.cpu().numpy().flatten()
            y_true = y_values  # (修复)
            r2 = r2_score(y_true, y_pred)

        return r2

    def predict(self, X):
        """
        预测价格
        """
        self.model.eval()

        with torch.no_grad():
            features = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(features)
            predictions = outputs.cpu().numpy().flatten()
            # .cpu(): 将预测结果从 GPU 移回 CPU  .numpy(): 将 PyTorch 张量转换为 NumPy 数组
            # .flatten(): 将 N行x1列 的数组展平为 1D 数组，返回最终的 1D 预测数组

        return predictions

    def _create_optimizer(self, config):
        """
        (新增) 根据配置动态创建优化器
        """
        optimizer_name = config.get('name', 'Adam').lower()
        lr = config.get('lr', 0.001)
        weight_decay = config.get('weight_decay', 0)

        print(f"创建优化器: {optimizer_name}, 学习率: {lr}, 权重衰减: {weight_decay}")

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer_name == 'sgd':
            momentum = config.get('momentum', 0.9)  # SGD 特有的动量参数
            print(f"  -> SGD 动量: {momentum}")
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        elif optimizer_name == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        else:
            print(f"警告: 未知的优化器 '{optimizer_name}'。将使用默认的 Adam。")
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, X_train, y_train, X_val, y_val,
              optimizer_config,  # (新增) 优化器配置字典
              batch_size=256,  # (修改) 从 training_config 传入
              epochs=500,  # (修改) 从 training_config 传入
              patience=20):  # (修改) 从 training_config 传入
        """
        训练神经网络模型
        """
        print(f"\n=== 开始模型训练 ===")
        print(f"Batch Size: {batch_size}, Epochs: {epochs}, Patience: {patience}")

        # 创建数据加载器
        train_loader = DataLoader(UsedCarDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        # shuffle=True（训练集） 在每个 epoch 开始时打乱数据
        val_loader = DataLoader(UsedCarDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        # 验证集不需要打乱

        # (修改) 动态定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = self._create_optimizer(optimizer_config)

        early_stop_counter = 0
        self.best_val_loss = float('inf')
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_r2_scores.clear()
        self.val_r2_scores.clear()

        # 开始 epochs 轮训练的主循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                # 清除上一批次的梯度
                optimizer.zero_grad()
                loss.backward()  # 计算损失函数相对于模型参数的梯度（反向传播）
                optimizer.step()  # 优化器根据梯度更新模型的权重
                train_loss += loss.item() * features.size(0)
                # 计算加权平均损失
            train_loss /= len(train_loader.dataset)

            # 验证模式
            self.model.eval()
            val_loss = 0.0
            y_pred_val = []
            y_true_val = []
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    val_loss += criterion(outputs, labels).item() * features.size(0)
                    y_pred_val.extend(outputs.cpu().numpy().flatten())
                    y_true_val.extend(labels.cpu().numpy().flatten())
            val_loss /= len(val_loader.dataset)

            # 计算 R² 分数
            train_r2 = self.evaluate(X_train, y_train)
            val_r2 = r2_score(y_true_val, y_pred_val)

            # 将当轮的指标保存到类属性列表中
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_r2_scores.append(train_r2)
            self.val_r2_scores.append(val_r2)

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"\n早停于第 {epoch + 1} 轮，验证集损失不再下降")
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # 这里的 return 语句已经是正确的，因为它返回了 self. 列表
        return {
            'train_losses': self.train_losses, 'val_losses': self.val_losses,
            'train_r2_scores': self.train_r2_scores, 'val_r2_scores': self.val_r2_scores,
            'best_val_loss': self.best_val_loss
        }