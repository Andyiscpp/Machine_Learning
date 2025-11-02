import torch.nn as nn

class PricePredictorModel(nn.Module):
    """
    二手车价格预测的神经网络模型结构。
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        #hidden_dims=[128, 64, 32]: 一个列表，定义了每个隐藏层的神经元数量。这里默认是三层，分别为 128、64、32 个神经元
        #dropout_rate=0.3: 设置 Dropout 层的丢弃率，防止过拟合
        """
        构建神经网络模型
        """
        super(PricePredictorModel, self).__init__()

        print(f"\n=== 构建神经网络模型 ===")

        layers = []#用来按顺序存放模型的所有层（如线性层、激活层等）
        prev_dim = input_dim#用于动态地将前一层的输出维度连接到下一层的输入维度

        # 添加隐藏层（包含线性层、批归一化、ReLU激活函数和Dropout）
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))#用于动态地将前一层的输出维度连接到下一层的输入维度
            layers.append(nn.ReLU())#为模型引入非线性，使其能够学习更复杂的关系
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
            #将 prev_dim 更新为当前层的输出维度 dim（例如 128），以便下一次循环（创建 64 维层时）可以将其用作输入维度

        # 输出层（预测一个价格值）
        layers.append(nn.Linear(prev_dim, 1))
        #回归任务（预测价格），所以输出维度是 1，并且不需要激活函数

        # 封装为Sequential模型
        self.model = nn.Sequential(*layers)

        print(f"模型结构:")
        print(self.model)

    def forward(self, x):
        return self.model(x)