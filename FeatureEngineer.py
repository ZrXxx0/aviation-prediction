import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from time_granularity import TimeGranularityController

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """数据预处理类，处理缺失值填充和归一化"""
    def __init__(self, fill_method='interp', max_invalid_ratio=1, min_fit_points=3, normalize=False):
        """
        Args:
            fill_method (str): 填充方法 ['interp'|'zero'|'regression']
            max_invalid_ratio (float): 最大允许缺失值比例
            min_fit_points (int): 回归填充所需的最小有效点数
            normalize (bool): 是否执行归一化
        """
        self.fill_method = fill_method
        self.max_invalid_ratio = max_invalid_ratio
        self.min_fit_points = min_fit_points
        self.normalize = normalize
        self.scaler = None
        self.binary_columns = []
        
    def fit(self, X, y=None):
        # 识别0-1列
        self.binary_columns = [
            col for col in X.columns 
            if set(X[col].dropna().unique()).issubset({0, 1})
        ]
        return self
        
    def transform(self, X):
        X_filled = X.copy()
        
        # 填充处理
        for col in X.columns:
            if col in self.binary_columns:
                continue  # 跳过0-1列
                
            ts = X[col].copy()
            invalid_mask = ts.isna() | (ts == 0)
            invalid_ratio = invalid_mask.mean()
            
            if invalid_ratio > self.max_invalid_ratio:
                continue
                
            ts[ts == 0] = np.nan
                
            if self.fill_method == 'interp':
                ts_filled = ts.interpolate(method='linear', limit_direction='both')
                ts_filled = self._handle_tail_missing(ts_filled)
                
            elif self.fill_method == 'regression':
                ts_filled = self._regression_fill(ts)
                
            elif self.fill_method == 'zero':
                ts_filled = ts.fillna(0)
                
            X_filled[col] = ts_filled
        
        # 归一化处理
        if self.normalize:
            non_binary_cols = [c for c in X.columns if c not in self.binary_columns]
            self.scaler = StandardScaler()
            X_filled[non_binary_cols] = self.scaler.fit_transform(X_filled[non_binary_cols])
            
        return X_filled

    def _handle_tail_missing(self, ts):
        """处理尾部缺失值"""
        if ts.isna().any():
            last_valid_idx = ts.last_valid_index()
            if last_valid_idx is not None and last_valid_idx < len(ts) - 1:
                recent_valid = []
                idx = last_valid_idx
                while idx >= 0 and not pd.isna(ts.iloc[idx]):
                    recent_valid.append((idx, ts.iloc[idx]))
                    idx -= 1
                
                if len(recent_valid) >= self.min_fit_points:
                    recent_valid = recent_valid[::-1]
                    x = np.arange(len(recent_valid))
                    y = np.array([v for _, v in recent_valid])
                    slope, intercept, *_ = stats.linregress(x, y)
                    
                    for i in range(last_valid_idx + 1, len(ts)):
                        ts.iloc[i] = slope * (i - recent_valid[0][0]) + intercept
        return ts

    def _regression_fill(self, ts):
        """回归填充整个序列"""
        # 实现略，根据实际特征工程需要调整
        return ts.interpolate(method='linear', limit_direction='both')

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """特征工程类，构建时间特征和统计特征"""
    def __init__(self, granularity_controller, lags=None, windows=None, 
                 holiday_months=None, add_ts_forecast=False, ts_model=None):
        """
        Args:
            granularity_controller (TimeGranularityController): 时间粒度控制器
            lags (list): 滞后阶数
            windows (list): 滑动窗口大小
            holiday_months (list): 假期月份
            add_ts_forecast (bool): 是否添加时间序列预测特征
            ts_model: 时间序列预测模型对象
        """
        self.granularity_controller = granularity_controller
        self.lags = lags or granularity_controller.get_lags()
        self.windows = windows or granularity_controller.get_windows()
        self.holiday_months = holiday_months or granularity_controller.get_holiday_months()
        self.target_col = 'Route_Total_Seats'
        self.add_ts_forecast = add_ts_forecast
        self.ts_model = ts_model
        self.fitted_ts_model = None
        
    def fit(self, X, y=None):
        # 如果需要添加时间序列预测特征，则训练时间序列模型
        # print(self.add_ts_forecast)
        if self.add_ts_forecast and self.ts_model is not None:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # 确保有日期列和目标列
            if 'YearMonth' in X.columns and self.target_col in X.columns:
                # 提取目标序列（按日期排序）
                ts_series = X.set_index('YearMonth')[self.target_col].sort_index()

                # # 转换索引为 DatetimeIndex
                # if not isinstance(ts_series.index, pd.DatetimeIndex):
                #     ts_series.index = pd.to_datetime(ts_series.index)
                
                # # 显式设置频率
                # if self.granularity_controller.get_freq():
                #     ts_series = ts_series.asfreq(self.granularity_controller.get_freq())

                # 克隆模型并训练
                self.fitted_ts_model = self.ts_model.__class__.__new__(self.ts_model.__class__)
                self.fitted_ts_model.__dict__ = self.ts_model.__dict__.copy()
                self.fitted_ts_model.fit(ts_series)
        return self
        
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = X.copy()
        
        if self.granularity_controller.granularity != 'yearly':
            X['Year'] = X['YearMonth'].dt.year
            X['Month'] = X['YearMonth'].dt.month
            X['Quarter'] = X['YearMonth'].dt.quarter
            X['Is_holiday'] = X['Month'].isin(self.holiday_months).astype(int)

        # 添加时间序列预测特征
        if self.add_ts_forecast and self.fitted_ts_model is not None and 'YearMonth' in X.columns:
            # 提取日期序列
            dates = X['YearMonth'].sort_values().unique()
            # 生成预测值
            try:
                # print(dates)
                ts_forecast = self.fitted_ts_model.predict(pd.DatetimeIndex(dates))
                # 对齐回原始数据
                # print(ts_forecast)
                forecast_series = pd.Series(ts_forecast.values, index=dates)
                # print(forecast_series)
                X['TS_Forecast'] = X['YearMonth'].map(forecast_series)
                # print(X['TS_Forecast'])
            except Exception as e:
                print(f"时间序列预测失败: {e}")
                # 回退到使用滞后特征
                X['TS_Forecast'] = X[self.target_col].shift(1)

        # 对于一条航线而言，这样的分箱其实没有意义！
        # # 距离和容量分箱，把航段距离按区间分为4档，把机型座位数按区间分为4档
        # if 'Distance_bin' not in X.columns:
        #     X['Distance_bin'] = pd.cut(
        #         X['Distance (KM)'],
        #         bins=[0, 800, 1500, 2000, np.inf],
        #         labels=['近程', '中程', '远程', '超远程']  # '近程(0-25%)', '中程(25-50%)', '远程(50-75%)', '超远程(75%+)'
        #     )
        # if not any(col.startswith(('Distance_bin_')) for col in X.columns):
        #     X = pd.get_dummies(X, columns=['Distance_bin'], dtype=int)
        
        # 滞后特征
        for lag in self.lags:
            X[f'{self.target_col}_lag_{lag}'] = X[self.target_col].shift(lag)
            # X[f'{self.target_col}_diff_{lag}'] = X[self.target_col] - X[f'{self.target_col}_lag_{lag}']
        
        # # 滑动特征
        # for window in self.windows:
        #     X[f'{self.target_col}_rollmean_{window}'] = X[self.target_col].rolling(window=window).mean()  # 这些都没必要
        
        return X

class AirlineRouteModel:
    """航线数据处理管道"""
    def __init__(self, data, preprocessor=None, feature_builder=None, granularity='monthly'):
        """
        Args:
            data (pd.DataFrame): 完整数据集
            preprocessor (DataPreprocessor): 数据预处理器
            feature_builder (FeatureBuilder): 特征构建器
            granularity (str): 时间粒度 ['monthly', 'quarterly', 'yearly']
        """
        self.data = data
        self.granularity_controller = TimeGranularityController(granularity)
        self.preprocessor = preprocessor or DataPreprocessor()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.date_col = 'YearMonth'
        self.target_col = 'Route_Total_Seats'
        
    def get_route_data(self, origin, destination):
        """获取特定航线数据"""
        route_data = self.data[
            (self.data['Origin'] == origin) & 
            (self.data['Destination'] == destination)
        ].copy()
        
        # 清理数据
        cols_to_drop = ['Origin', 'Destination', 'Equipment', 'International Flight',
                       'Equipment_Total_Flights', 'Equipment_Total_Seats', 'Route_Total_Flight_Time', 'Region',
                       'Route_Total_Flights', 'Con Total Est. Pax', 'First', 'Business',
                       'Premium', 'Full Y', 'Disc Y', 'Total Est. Pax', 'Local Est. Pax',
                       'Behind Est. Pax', 'Bridge Est. Pax', 'Beyond Est. Pax']
        
        route_data = (route_data.drop(columns=cols_to_drop, errors='ignore')
                                .drop_duplicates()
                                .query("YearMonth not in ['2024-06', '2024-07']")  # 这个得想办法解决了！
                                .assign(YearMonth=lambda df: pd.to_datetime(df['YearMonth']))
                                .sort_values(self.date_col)
                                .reset_index(drop=True)
        )
        route_data = self.granularity_controller.resample_data(route_data)
         # 移除最后可能不完整的时间段
        if self.granularity_controller.granularity != 'monthly':
            # 获取最新完整时间段的截止日期
            last_complete_date = route_data[self.date_col].max()
            route_data = route_data[route_data[self.date_col] < last_complete_date]
        
        return route_data
    
    def prepare_data(self, origin, destination, test_size=12):
        """准备训练/测试数据"""
        # 获取航线数据
        data = self.get_route_data(origin, destination)

        # 调整测试集大小（按粒度转换）
        if self.granularity_controller.granularity == 'quarterly':
            test_size = max(2*3, test_size)  # 至少2个季度 还是按照月度走
        elif self.granularity_controller.granularity == 'yearly':
            test_size = max(1, test_size // 12)  # 至少1年
        
        # 数据预处理
        data_preprocessed = self.preprocessor.fit_transform(data)
        
        # 特征工程 - 先fit再transform
        self.feature_builder.fit(data_preprocessed)
        data_with_features = self.feature_builder.transform(data_preprocessed)
        
        # 分割数据集
        test_start_date = data[self.date_col].max() - pd.DateOffset(months=test_size-1)
        train_data = data_with_features[data_with_features[self.date_col] < test_start_date]
        test_data = data_with_features[data_with_features[self.date_col] >= test_start_date]
        
        # 准备特征和标签
        exclude_cols = [self.date_col, self.target_col]
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        X_train = train_data[feature_cols]
        y_train = train_data[self.target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[self.target_col]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        return X_train, y_train, X_test, y_test, data_with_features

