import matplotlib.pyplot as plt #这是 Python 中最常用的绘图库
import os

class ResultsVisualizer:
    """
    结果可视化类，用于绘制和保存训练过程及预测结果的图表。
    """

    def __init__(self, trainer):
        self.trainer = trainer

    def visualize_results(self, X_test, y_test, save_dir='results'):    #接收测试数据集 X_test, y_test 和一个保存目录 save_dir
        """
        可视化预测结果
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 预测测试集
        y_pred = self.trainer.predict(X_test)

        # 1. 绘制真实值vs预测值散点图
        plt.figure(figsize=(10, 6)) #创建一个新的图表画布，设置大小
        plt.scatter(y_test, y_pred, alpha=0.5)  #散点图，x 轴是真实值 y_test，y 轴是预测值 y_pred。alpha=0.5 设置透明度
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        #找到 x 和 y 轴的最小值和最大值
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        #绘制一条红色的虚线对角线（y=x）。如果散点都在这条线附近，说明模型预测得越准。
        plt.xlabel('True price')
        plt.ylabel('Predicted price')
        plt.title('True price vs predicted price')
        plt.tight_layout()  #防止标签重叠
        plt.savefig(os.path.join(save_dir, 'true_vs_pred.png'))
        plt.close()

        # 2. 绘制预测误差分布直方图
        errors = y_pred - y_test.values
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7)
        #绘制误差的直方图（柱状图），bins=50 表示将数据分成 50 个“桶”来统计频数。
        # 一个好的模型，误差分布应该接近于 0，呈正态分布（钟形）
        plt.title('Prediction error distribution')
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()

        # 3. 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.trainer.train_losses, label='train loss')
        #从 self.trainer 对象中获取 train_losses 列表（训练过程中每个 epoch 的损失）并绘制曲线
        plt.plot(self.trainer.val_losses, label='test loss')
        plt.title('train and test loss curves')
        plt.legend()    #显示图例
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()

        # 4. 绘制R²分数曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.trainer.train_r2_scores, label='Train R²')
        plt.plot(self.trainer.val_r2_scores, label='Test R²')
        plt.title('Train and Test curves')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'r2_curves.png'))
        plt.close()

        print(f"可视化结果已保存至: {save_dir}")
        return y_pred