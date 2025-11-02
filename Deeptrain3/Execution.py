import PricePredictionWork as pipeline_module
import os


class ModelRunner:
    """
    独立运行类：负责配置并启动 UsedCarPricePipeline。
    这个类作为整个项目的入口点，不包含任何数据处理或模型逻辑。
    """

    def __init__(self, data_path, random_state=42, save_results=True, optimizer_config=None, training_config=None):
        """
        初始化运行器，并实例化主流程协调器。

        Args:
            data_path (str): 训练数据文件的路径。
            random_state (int): 随机种子。
            save_results (bool): 是否保存结果文件和图表。
            optimizer_config (dict): 优化器配置 (新增)。
            training_config (dict): 训练超参数配置 (新增)。
        """
        self.data_path = data_path
        self.random_state = random_state
        self.save_results = save_results

        # (新增) 存储配置
        self.optimizer_config = optimizer_config
        self.training_config = training_config

        # 正确调用：模块名.类名
        # (修改) 将配置传递给 Pipeline
        self.pipeline = pipeline_module.UsedCarPricePipeline(
            random_state=self.random_state,
            optimizer_config=self.optimizer_config,
            training_config=self.training_config
        )

    def run(self):
        """
        执行完整的训练和预测流程。
        """
        print(f"==========================================")
        print(f"      启动模型训练和预测流程 (ModelRunner)  ")
        print(f"==========================================")
        print(f"数据文件路径: {self.data_path}")
        print(f"优化器配置: {self.optimizer_config}")
        print(f"训练配置: {self.training_config}")

        if not os.path.exists(self.data_path):
            print(f"错误：数据文件未找到: {self.data_path}")
            return None

        try:
            # 调用 UsedCarPricePipeline 类启动整个流程
            # (修改) run_pipeline 不再需要传入参数，因为已在 init 中配置
            results = self.pipeline.run_pipeline(
                data_path=self.data_path,
                save_results=self.save_results
            )

            self._report_results(results)
            return results
        except Exception as e:
            print(f"\n流程执行失败，发生错误：{e}")
            # 如果需要，可以在这里添加更复杂的错误处理
            return None

    def _report_results(self, results):
        """
        打印最终的关键评估指标。
        """
        metrics = results.get('metrics', {})
        print(f"\n==========================================")
        print(f"         流程执行完毕 - 最终指标        ")
        print(f"==========================================")
        print(f"测试集 R² (决定系数): {metrics.get('test_r2', 'N/A'):.4f}")
        print(f"测试集 MSE (均方误差): {metrics.get('test_mse', 'N/A'):.2f}")
        print(f"最佳验证集损失: {metrics.get('best_val_loss', 'N/A'):.4f}")

        if self.save_results:
            print("\n训练模型、指标和图表已保存到最新生成的结果文件夹中。")
        print("==========================================")


# 程序的入口点：运行 ModelRunner
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    data_file = 'used_car_train.csv'

    # ----------------------------------------------------------------------
    # ======= (核心修改) 在此处灵活配置优化策略 =======
    #
    # 策略 1: Adam (您当前使用的)
    # Adam 是一种自适应学习率的优化器，通常效果很好，收敛较快。
    opt_config = {
        'name': 'Adam',
        'lr': 0.001,  # 学习率
        'weight_decay': 1e-5  # L2 正则化 (权重衰减)
    }

    # # 策略 2: SGD + Momentum (动量策略)
    # # SGD 配合动量（Momentum）可以帮助加速收敛并越过局部最小值。
    # # 它通常需要比 Adam 更仔细的调参（尤其是学习率）。
    # opt_config = {
    #     'name': 'SGD',
    #     'lr': 0.01,
    #     'momentum': 0.9,      # 动量因子
    #     'weight_decay': 1e-5
    # }
    #
    # # 策略 3: Adagrad
    # # Adagrad 也是一种自适应学习率优化器，它对稀疏数据很有效，但学习率会随时间单调递减。
    # opt_config = {
    #     'name': 'Adagrad',
    #     'lr': 0.01,
    #     'weight_decay': 1e-5
    # }
    # ----------------------------------------------------------------------
    # ======= (核心修改) 在此处灵活配置训练参数 (Batch 策略等) =======

    train_config = {
        'batch_size': 128,  # Batch 策略: 批量大小
        'epochs': 500,  # 总训练轮数
        'patience': 40  # 早停耐心值
    }
    # ----------------------------------------------------------------------

    runner = ModelRunner(
        data_path=data_file,
        random_state=42,
        save_results=True,
        optimizer_config=opt_config,  # (新增) 传入优化器配置
        training_config=train_config  # (新增) 传入训练配置
    )
    final_results = runner.run()