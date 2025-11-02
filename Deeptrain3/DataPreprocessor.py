import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    二手车数据预处理类，负责加载、清洗、特征工程、标准化和数据集划分。
    """

    def __init__(self, random_state=42, target_col='price'):
        self.random_state = random_state
        self.target_col = target_col
        self.scaler = None  #初始化一个占位符，用于存储 StandardScaler 对象
        self.feature_names = None
        self.outlier_thresholds = {}    #初始化一个字典，用于存储在异常值检测过程中计算出的各特征的上下界

    def load_data(self, file_path):
        """
        加载数据文件
        """
        print(f"加载数据文件: {file_path}")
        try:
            df = pd.read_csv(file_path)
            print(f"数据加载成功，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise

    def detect_outliers_iqr(self, series, column_name, threshold=1.5):
        #iqr是一种标准的统计学方法，用于识别异常值
        #series: 接收一个 Pandas Series（即 DataFrame 的某一列）。
        #threshold=1.5: 这是 IQR 方法的标准阈值。
        """
        使用IQR方法检测异常值
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (series < lower_bound) | (series > upper_bound)

        print(f"{column_name} 异常值检测:")
        print(f"  下界: {lower_bound:.2f}, 上界: {upper_bound:.2f}")
        print(f"  异常值数量: {outliers.sum()}, 比例: {outliers.sum() / len(series):.2%}")

        self.outlier_thresholds[column_name] = (lower_bound, upper_bound)
        return outliers, lower_bound, upper_bound

    def preprocess_data(self, df, test_size=0.2, val_size=0.125):
        # 这个 0.125 是相对于剩余的 80% 数据（X_train_val）而言的。
        # (0.8 * 0.125 = 0.1)，所以最终划分是：70% 训练, 10% 验证, 20% 测试
        """
        数据预处理，包含异常值处理和特征工程
        """
        print("\n=== 开始数据预处理 ===")
        df_clean = df.copy()    #创建 DataFrame 的一个副本，避免修改原始传入的 df

        # 1. 处理目标变量price的异常值
        print("\n1. 处理目标变量price异常值")
        price_outliers, price_lower, price_upper = self.detect_outliers_iqr(df_clean[self.target_col], self.target_col)
        # 对价格异常值进行截断处理
        df_clean.loc[df_clean[self.target_col] > price_upper, self.target_col] = price_upper
        df_clean.loc[df_clean[self.target_col] < price_lower, self.target_col] = price_lower

        # 2. 处理object类型列
        # 2.1 处理gearbox列: 清洗脏数据并填充缺失值
        print("2.1 处理 gearbox 列 (清洗脏数据并填充)")

        # 步骤 2.1.1: 将非数字（如 '-'）转为 NaN，保留数字（0, 1, 136 等）
        df_clean['gearbox'] = pd.to_numeric(df_clean['gearbox'], errors='coerce')

        # 步骤 2.1.2: 找出有效的 0 和 1，并计算它们的众数
        #         我们只在有效值（0 或 1）中计算众数，忽略 NaN 和脏数据（如 136）
        valid_gearbox = df_clean.loc[df_clean['gearbox'].isin([0, 1]), 'gearbox']
        # 如果 valid_gearbox 为空（虽然不太可能），则默认填充 0 或 1，这里选 0
        if valid_gearbox.empty:
            mode_val = 0
        else:
            mode_val = valid_gearbox.mode()[0]

        print(f"  gearbox 列有效值的众数为: {mode_val}")

        # 步骤 2.1.3: 将所有无效值（NaN 或 非 0/1 的数字如 136）替换为计算出的众数
        #         首先填充 NaN
        df_clean['gearbox'].fillna(mode_val, inplace=True)
        #         然后将非 0/1 的值也设为众数
        df_clean.loc[~df_clean['gearbox'].isin([0, 1]), 'gearbox'] = mode_val

        # 步骤 2.1.4: 确保数据类型为整数
        df_clean['gearbox'] = df_clean['gearbox'].astype(int)

        print(f"  gearbox 列处理后值分布:\n{df_clean['gearbox'].value_counts()}")

        # 2.2 处理power列: 限制范围并填充缺失值
        # 步骤 2.2.1: 将 '-' 替换为 NaN，以便后续统一处理缺失值
        df_clean['power'] = df_clean['power'].replace('-', np.nan)

        # 步骤 2.2.2: 强制转换为数值，无法转换的值（包括上一步的NaN）变为 NaN
        df_clean['power'] = pd.to_numeric(df_clean['power'], errors='coerce')

        # 步骤 2.2.3: 应用新的硬编码边界
        #   - 将所有大于 18000 的值设为 18000
        df_clean.loc[df_clean['power'] > 600, 'power'] = 600
        #   - 将所有小于等于 0 的值设为 5
        df_clean.loc[df_clean['power'] <= 0, 'power'] = 15

        # 步骤 2.2.4: 用处理边界后的列的中位数填充所有剩余的 NaN 值
        # 注意：现在中位数是在边界处理之后的数据上计算的
        power_median = df_clean['power'].median()
        df_clean['power'].fillna(power_median, inplace=True)
        print(f"  power 列处理后，使用中位数 {power_median} 填充 NaN")
        print(f"  power 列处理后最小值: {df_clean['power'].min()}, 最大值: {df_clean['power'].max()}")

        # 2.3 处理kilometer列: 清洗、限制范围(0-15)并填充缺失值
        print("2.3 处理 kilometer 列 (清洗, 边界: 0-15)")

        # 步骤 2.3.1: 将 '-' (或其他可能的非数值符号) 替换为 NaN
        df_clean['kilometer'] = df_clean['kilometer'].replace('-', np.nan)  # 明确处理'-'

        # 步骤 2.3.2: 强制转换为数值，无法转换的值（包括NaN）变为 NaN
        df_clean['kilometer'] = pd.to_numeric(df_clean['kilometer'], errors='coerce')

        # 步骤 2.3.3: 应用基于数据分析的硬编码边界
        #   - 将所有大于 15.0 的值设为 15.0 (基于数据的实际最大值)
        df_clean.loc[df_clean['kilometer'] > 15.0, 'kilometer'] = 15.0
        #   - 将所有小于 0 的值设为 0 (保留健壮性)
        df_clean.loc[df_clean['kilometer'] < 0, 'kilometer'] = 0

        # 步骤 2.3.4: 用处理边界后的列的中位数填充所有剩余的 NaN 值
        km_median = df_clean['kilometer'].median()
        df_clean['kilometer'].fillna(km_median, inplace=True)
        print(f"  kilometer 列处理后，使用中位数 {km_median} 填充 NaN")
        print(f"  kilometer 列处理后最小值: {df_clean['kilometer'].min()}, 最大值: {df_clean['kilometer'].max()}")

        # 2.4 处理notRepairedDamage列: 转换为数值并填充为0
        # 步骤 2.4.1: 将 '-' 替换为 NaN，并将列转为数值
        df_clean['notRepairedDamage'] = df_clean['notRepairedDamage'].replace('-', np.nan)
        # errors='coerce' 会把无法转换的（比如万一还有其他文本）变成NaN
        df_clean['notRepairedDamage'] = pd.to_numeric(df_clean['notRepairedDamage'], errors='coerce')

        # 步骤 2.4.2: 创建新的 'damage_cost' (损伤价格) 特征
        # 我们做一个假设：NaN (信息缺失) 等同于 价格为 0
        df_clean['damage_cost'] = df_clean['notRepairedDamage'].fillna(0)

        # 步骤 2.4.3: 创建新的 'has_damage' (是否有损伤) 二元特征
        # 只要 cost > 0，就代表 '有损伤' (1)，否则为 '无损伤' (0)
        df_clean['has_damage'] = (df_clean['damage_cost'] > 0).astype(int)

        # 步骤 2.4.4: (可选) 检查 'damage_cost' 的异常值
        # 因为这是一个价格，我们可以像处理 'power' 列一样，对其进行异常值检测或截断
        # (为简化起见，这里暂时省略，但您可以在此添加 detect_outliers_iqr)

        # 步骤 2.4.5: 删除原始的 'notRepairedDamage' 列
        # 我们现在有了 'damage_cost' 和 'has_damage' 两个新特征
        df_clean = df_clean.drop('notRepairedDamage', axis=1)

        # === 新增代码块：处理分类特征的缺失值 ===
        print("\n2.5 处理 brand, bodyType, fuelType, model 的缺失值")
        categorical_cols = ['brand', 'bodyType', 'fuelType', 'model']
        for col in categorical_cols:
            if col in df_clean.columns:  # 确保列存在
                # 检查是否存在 NaN 或潜在的非数值（如果之前没处理过）
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')  # 确保是数值或NaN

                if df_clean[col].isnull().any():  # 如果确实有缺失值
                    # 计算众数
                    mode_val = df_clean[col].mode()[0]
                    # 使用众数填充 NaN
                    df_clean[col].fillna(mode_val, inplace=True)
                    print(f"  {col} 列的 NaN 值已用众数 {mode_val} 填充")
                    # (可选) 确保是整数类型
                    # df_clean[col] = df_clean[col].astype(int)
            else:
                print(f"  警告: 列 {col} 不在 DataFrame 中，跳过处理")
        # === 新增代码块结束 ===

        # 3. 处理其他缺失值: 使用中位数填充
        missing_cols = df_clean.columns[df_clean.isnull().sum() > 0]
        for col in missing_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

        # 4. 特征选择: 移除无用列
        drop_cols = ['SaleID', 'name', 'regDate', 'creatDate']
        df_clean = df_clean.drop(drop_cols, axis=1)#axis=1 表示删除的是“列”

        # 分离特征和目标变量
        #X 是 df_clean 中除了目标列之外的所有列，y是目标列
        X = df_clean.drop(self.target_col, axis=1)
        y = df_clean[self.target_col]
        self.feature_names = X.columns.tolist()

        # 5. 特征标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        #fit：Scaler "学习" 训练数据 X 中每一列的均值 (mean) 和标准差 (std)。
        #transform：Scaler 使用刚刚学到的均值和标准差，对 X 中的每一列应用公式 (value - mean) / std
        #将数据转换为标准化形态。神经网络对特征的尺度非常敏感，标准化（或归一化）是必须的步骤

        # 6. 数据集划分
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=self.random_state
        )
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

        print(f"数据预处理后，训练集形状: {X_train.shape}")
        print(f"数据预处理后，验证集形状: {X_val.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test