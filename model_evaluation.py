import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

class ModelEvaluator:
    """
    模型评估器，封装时间序列预测评估指标
    
    Attributes:
        y_true (array): 真实值数组
        y_pred (array): 预测值数组
        metrics (dict): 存储计算后的指标
    """
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = {}

    # @staticmethod
    # def smape(y_true, y_pred):
    #     """计算对称平均绝对百分比误差"""
    #     return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    # @staticmethod
    # def mase(y_true, y_pred, period=12):
    #     """计算比例平均绝对误差"""
    #     naive_error = np.mean(np.abs(y_true[period:] - y_true[:-period]))
    #     return mean_absolute_error(y_true, y_pred) / naive_error

    def calculate_metrics(self):
        """计算全部评估指标"""
        self.metrics = {
            'RMSE': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'MAPE': mean_absolute_percentage_error(self.y_true, self.y_pred),
            'R²': r2_score(self.y_true, self.y_pred)
        }
        return self

    def report(self, name="Test"):
        """生成评估报告"""
        print(f"\n{name} Metrics:")
        for k, v in self.metrics.items():
            if k in ['MAPE', 'sMAPE']:
                print(f"{k}: {v:.2%}")
            elif k == 'R²':
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v:.2f}")