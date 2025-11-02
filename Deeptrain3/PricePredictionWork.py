import pandas as pd
import numpy as np
# pandas 和 numpy: 用于数据处理和数值计算
import torch
# torch: PyTorch 库，用于构建和训练神经网络
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import json
from datetime import datetime

from Deeptrain3.NumpyEncoder import NumpyEncoder
from Deeptrain3.DataPreprocessor import DataPreprocessor
from Deeptrain3.ModelManager import ModelTrainer
from Deeptrain3.ModelArchitecture import PricePredictorModel
from Deeptrain3.ResultsVisualizer import ResultsVisualizer


class UsedCarPricePipeline:
    """
    二手车价格预测的完整流程类，协调数据预处理、模型训练、保存和可视化。
    """

    def __init__(self, random_state=42, optimizer_config=None, training_config=None):
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # (新增) 存储配置
        # 如果未提供配置，则使用默认值
        self.optimizer_config = optimizer_config if optimizer_config is not None else {
            'name': 'Adam', 'lr': 0.001, 'weight_decay': 1e-5
        }
        self.training_config = training_config if training_config is not None else {
            'batch_size': 256, 'epochs': 500, 'patience': 20
        }

        # 组件实例
        self.preprocessor = DataPreprocessor(random_state=random_state)
        self.model = None
        self.trainer = None  # Trainer 会在 run_pipeline 时根据模型动态创建
        self.visualizer = None

        self.X_test = None
        self.y_test = None

    def save_components(self, save_dir='models'):
        """
        保存模型和相关组件
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # (修改) 确保 trainer 已经初始化
        if self.trainer is None or self.trainer.best_model_state is None:
            print("错误：模型尚未训练，无法保存。")
            return

        # 保存模型、标准化器和特征名称
        torch.save(self.trainer.best_model_state, os.path.join(save_dir, 'car_price_model.pth'))
        # 保存 self.trainer 对象中存储的 best_model_state（即验证集上表现最好的模型权重)
        joblib.dump(self.preprocessor.scaler, os.path.join(save_dir, 'scaler.joblib'))
        # 保存 self.preprocessor 对象中的 scaler（即 StandardScaler 标准化器）,用于将特征数据标准化.可避免特征偏差
        with open(os.path.join(save_dir, 'feature_names.txt'), 'w') as f:
            f.write('\n'.join(self.preprocessor.feature_names))

        print(f"模型组件已保存至: {save_dir}")  # 即models文件夹下

    def load_components(self, model_dir='models'):
        """
        加载模型和相关组件
        """
        # 加载特征名称以确定模型输入维度
        features_path = os.path.join(model_dir, 'feature_names.txt')
        with open(features_path, 'r') as f:
            self.preprocessor.feature_names = [line.strip() for line in f.readlines()]
            input_dim = len(self.preprocessor.feature_names)
            # 明确需要的输入特征的数量

        # 初始化模型和训练器
        self.model = PricePredictorModel(input_dim=input_dim)
        # (修改) 传入模型和配置来初始化训练器
        self.trainer = ModelTrainer(self.model, self.random_state)

        # 加载模型状态
        model_path = os.path.join(model_dir, 'car_price_model.pth')
        # (修改) 确保设备正确
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)  # 确保模型在正确的设备上
        # 将加载的权重应用到 self.model 的结构中。
        self.model.eval()
        # 将模型切换到“评估模式”（evaluation mode）。这会关闭 Dropout 和 BatchNorm 的训练行为，确保预测结果是确定的。

        # 加载标准化器
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        self.preprocessor.scaler = joblib.load(scaler_path)

        print(f"模型组件已从 {model_dir} 加载")

    def run_pipeline(self, data_path, save_results=True):
        """
        运行完整的预测流程
        """
        # 1. 数据预处理
        df = self.preprocessor.load_data(data_path)
        X_train, X_val, self.X_test, y_train, y_val, self.y_test = self.preprocessor.preprocess_data(df)
        # preprocess_data 方法执行所有数据清洗、特征工程、分割和标准化

        # 2. 模型构建和训练
        input_dim = self.X_test.shape[1]
        self.model = PricePredictorModel(input_dim=input_dim)

        # (修改) 将模型和配置传递给 ModelTrainer
        self.trainer = ModelTrainer(self.model, random_state=self.random_state)

        # (修改) 将配置字典解包后传入 train 方法
        training_history = self.trainer.train(
            X_train, y_train, X_val, y_val,
            optimizer_config=self.optimizer_config,
            **self.training_config  # 解包 batch_size, epochs, patience
        )
        # 一个包含损失和 R² 历史记录的 training_history 字典

        # 3. 评估模型
        y_pred = self.trainer.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_pred)
        test_r2 = r2_score(self.y_test, y_pred)
        # self.trainer 内部保存了训练好的最佳模型。这里调用其 predict 方法对测试集 self.X_test 进行预测

        # 4. 准备结果
        results = {
            'metrics': {
                'test_mse': test_mse,
                'test_r2': test_r2,
                'best_val_loss': training_history['best_val_loss'],
                'final_train_r2': training_history['train_r2_scores'][-1] if training_history[
                    'train_r2_scores'] else None,
                'final_val_r2': training_history['val_r2_scores'][-1] if training_history['val_r2_scores'] else None
            },
            'predictions': pd.DataFrame({'true_price': self.y_test, 'predicted_price': y_pred}),
            'training_history': training_history
        }

        # 5. 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = f'results_{timestamp}'

            self.visualizer = ResultsVisualizer(self.trainer)
            self.save_components(os.path.join(results_dir, 'models'))
            self.visualizer.visualize_results(self.X_test, self.y_test, os.path.join(results_dir, 'figures'))

            # 保存 CSV 和 JSON
            results['predictions'].to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
            with open(os.path.join(results_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(results['metrics'], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        return results


# 运行示例（在实际部署时，您只需运行以下部分）
if __name__ == "__main__":
    # (修改) 展示如何使用新配置进行初始化

    # 示例1：使用默认 Adam
    # predictor_pipeline = UsedCarPricePipeline(random_state=42)

    # 示例2：自定义 SGD+Momentum
    # opt_config_sgd = {'name': 'SGD', 'lr': 0.01, 'momentum': 0.9}
    # train_config_custom = {'batch_size': 512, 'epochs': 100, 'patience': 10}

    # predictor_pipeline = UsedCarPricePipeline(
    #     random_state=42,
    #     optimizer_config=opt_config_sgd,
    #     training_config=train_config_custom
    # )

    # results = predictor_pipeline.run_pipeline('used_car_train.csv', save_results=True)

    print("\n--- 完整的解耦流程已定义。请使用 UsedCarPricePipeline 类的 run_pipeline 方法启动预测任务。---")