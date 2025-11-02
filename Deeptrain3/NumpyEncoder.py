# 导入必要的库
# 确保在将评估指标保存为 JSON 文件时，NumPy 数据类型（如 np.float32）可以被正确地序列化。
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """
    处理NumPy数据类型的JSON编码器，以便正确保存指标文件。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):#检查对象是否是 NumPy 数组 (
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)