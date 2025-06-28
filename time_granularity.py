import pandas as pd

class TimeGranularityController:
    """控制不同时间粒度的数据处理"""
    def __init__(self, granularity='monthly'):
        """
        Args:
            granularity (str): 时间粒度 ['monthly', 'quarterly', 'yearly']
        """
        self.granularity = granularity
        self.granularity_map = {
            'monthly': 'MS',
            'quarterly': 'QS',
            'yearly': 'YS'
        }
    
    def resample_data(self, df, date_col='YearMonth'):
        """按指定粒度聚合数据"""
        if self.granularity == 'monthly':
            return df
        
        # 设置日期索引
        # print(df.columns)
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # 定义聚合规则
        agg_rules = {
            'Route_Total_Seats': 'sum',
            'Distance (KM)': 'mean',
            'Route_Avg_Flight_Time':'mean',
            'Avg yield':'mean', 
            'Avg First':'mean', 
            'Avg Business':'mean',
            'Avg Premium':'mean', 
            'Avg Full Y':'mean', 
            'Avg Disc Y':'mean', 
            'Avg Fare (USD)':'mean',
            'Local Fare':'mean', 
            'Behind Fare':'mean', 
            'Bridge Fare':'mean', 
            'Beyond Fare':'mean', 
            'O_GDP':'mean',
            'O_Population':'mean', 
            'Third_Industry_x':'mean', 
            'O_Revenue':'mean', 
            'O_Retail':'mean', 
            'O_Labor':'mean',
            'O_Air_Traffic':'mean', 
            'D_GDP':'mean', 
            'D_Population':'mean', 
            'Third_Industry_y':'mean',
            'D_Revenue':'mean', 
            'D_Retail':'mean', 
            'D_Labor':'mean', 
            'D_Air_Traffic':'mean'
        }
        
        # 执行重采样
        resampled = df.resample(self.granularity_map[self.granularity]).agg(agg_rules)
        resampled[date_col] = resampled.index
        return resampled.reset_index(drop=True)
    
    def get_lags(self):
        """根据粒度返回合适的滞后阶数"""
        return {
            'monthly': [1, 3, 6],
            'quarterly': [1, 2, 4],
            'yearly': [1, 2]
        }[self.granularity]
    
    def get_windows(self):
        """根据粒度返回合适的滑动窗口大小"""
        return {
            'monthly': [3, 6, 12],
            'quarterly': [2, 4],
            'yearly': [2]
        }[self.granularity]
    
    def get_holiday_months(self):
        """根据粒度调整假期月份"""
        return {
            'monthly': [1, 2, 7, 8],
            'quarterly': [1],  # Q1包含春节
            'yearly': []  # 年度预测不需要月度假期
        }[self.granularity]
    
    def get_freq(self):
        """获取时间频率字符串"""
        # print(self.granularity_map[self.granularity])
        return self.granularity_map[self.granularity]