import pandas as pd
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import Holt
import warnings

from time_granularity import TimeGranularityController


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """智能数据预处理类，区分经济数据列和其他列使用不同的尾部填充策略"""
    def __init__(self, fill_method='interp', max_invalid_ratio=1, min_fit_points=5, 
                 normalize=False, time_col='YearMonth',
                 non_economic_tail_window=3,
                 non_economic_model='sarima', forecast_horizon=100,
                 economic_prefixes=(
                    'O_GDP','O_Population','Third_Industry_x','O_Revenue','O_Retail','O_Labor','O_Air_Traffic',
                    'D_GDP','D_Population','Third_Industry_y','D_Revenue','D_Retail','D_Labor','D_Air_Traffic'
                 )):
        """
        Args:
            fill_method (str): 主填充方法 ['interp'|'zero'|'regression']
            max_invalid_ratio (float): 最大允许缺失值比例
            min_fit_points (int): 回归填充所需的最小有效点数
            normalize (bool): 是否执行归一化
            time_col (str): 时间列名称
            non_economic_model (str): 非经济数据列预测模型 ['ses'|'linear'|'lastn']
            forecast_horizon (int): 最大预测步长
            non_economic_tail_window (int): 非经济数据列尾部填充使用的窗口大小（月数）
            economic_prefixes (tuple): 经济数据列的前缀列表
        """
        self.fill_method = fill_method
        self.max_invalid_ratio = max_invalid_ratio
        self.min_fit_points = min_fit_points
        self.normalize = normalize
        self.time_col = time_col
        self.economic_prefixes = economic_prefixes
        self.non_economic_tail_window = non_economic_tail_window
        self.non_economic_model = non_economic_model
        self.forecast_horizon = forecast_horizon
        self.scaler = None
        self.binary_columns = []
        self.column_stats = {}
        self.economic_columns = []
        self.model_params={'auto': {'seasonal': True, 'seasonal_periods': 12},
                              'arima': {'order': (1,1,1)},
                              'sarima': {'order': (1,1,1), 'seasonal_order': (1,1,1,12)}}
        
    def fit(self, X, y=None):
        # 识别0-1列
        self.binary_columns = [
            col for col in X.columns 
            if col != self.time_col and set(X[col].dropna().unique()).issubset({0, 1})
        ]
        
        # 识别经济数据列
        self.economic_columns = [
            col for col in X.columns 
            if col != self.time_col and any(col.startswith(prefix) for prefix in self.economic_prefixes)
        ]
        
        # 收集列统计信息用于后备填充
        for col in X.columns:
            if col != self.time_col and col not in self.binary_columns:
                non_zero = X[col][(X[col] != 0) & (X[col].notna())]
                if len(non_zero) > 0:
                    self.column_stats[col] = {
                        'mean': non_zero.mean(),
                        'median': non_zero.median(),
                        'min': non_zero.min(),
                        'max': non_zero.max()
                    }
        return self
        
    def transform(self, X):
        # 深拷贝数据并确保时间列存在
        if self.time_col not in X.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in data")
            
        X_filled = X.copy()
        time_series = pd.to_datetime(X_filled[self.time_col])
        
        # 第一步：将0值转换为NaN（0-1列除外）
        for col in X.columns:
            if col != self.time_col and col not in self.binary_columns:
                X_filled.loc[X_filled[col] == 0, col] = np.nan
        
        # 第二步：按列进行填充处理
        for col in X.columns:
            if col in self.binary_columns or col == self.time_col:
                continue
                
            ts = X_filled[col].copy()
            original_na_count = ts.isna().sum()
            
            if ts.isna().mean() > self.max_invalid_ratio:
                # 缺失过多，使用后备填充
                if col in self.column_stats:
                    ts.fillna(self.column_stats[col]['median'], inplace=True)
                else:
                    ts.fillna(0, inplace=True)
                X_filled[col] = ts
                continue
                
            # 处理头部缺失
            first_valid_idx = ts.first_valid_index()
            if first_valid_idx is not None:
                # 使用第一个有效值向前填充头部
                first_valid_value = ts.loc[first_valid_idx]
                ts.loc[:first_valid_idx] = first_valid_value
            else:
                # 无有效值，使用列统计
                if col in self.column_stats:
                    ts.fillna(self.column_stats[col]['median'], inplace=True)
                else:
                    ts.fillna(0, inplace=True)
                X_filled[col] = ts
                continue
                
            # 主填充方法
            if self.fill_method == 'interp':
                # 仅填充中间缺失，保留尾部缺失
                ts = self._safe_interpolate(ts, time_series)
            elif self.fill_method == 'regression':
                ts = self._regression_fill(ts, time_series)
            elif self.fill_method == 'zero':
                ts.fillna(0, inplace=True)
            
            # 专门处理尾部缺失 - 根据列类型使用不同策略
            ts = self._fill_tail_missing(ts, time_series, col)
                
            # 最终后备填充（处理任何剩余缺失）
            if ts.isna().any():
                if col in self.column_stats:
                    ts.fillna(self.column_stats[col]['median'], inplace=True)
                else:
                    # 线性插值作为最后手段
                    ts.interpolate(method='linear', limit_direction='both', inplace=True)
            
            X_filled[col] = ts
        
        # 0-1列特殊处理
        for col in self.binary_columns:
            if X_filled[col].isna().any():
                mode_val = X_filled[col].mode()
                if not mode_val.empty:
                    X_filled.loc[:, col] = X_filled[col].fillna(mode_val[0])
                else:
                    X_filled.loc[:, col] = X_filled[col].fillna(0)
        
        # 归一化处理
        if self.normalize:
            non_binary_cols = [c for c in X.columns if c not in self.binary_columns and c != self.time_col]
            if non_binary_cols:
                self.scaler = StandardScaler()
                X_filled[non_binary_cols] = self.scaler.fit_transform(X_filled[non_binary_cols])
            
        return X_filled

    def _safe_interpolate(self, ts, time_series):
        """安全插值，避免填充尾部"""
        # 创建临时副本用于插值
        ts_temp = ts.copy()
        
        # 标记尾部缺失位置
        last_valid_idx = ts_temp.last_valid_index()
        if last_valid_idx and last_valid_idx < len(ts_temp) - 1:
            tail_mask = ts_temp.index > last_valid_idx
            ts_temp[tail_mask] = np.nan  # 确保尾部保持为NaN
        
        # 执行插值（仅限非尾部）
        ts_temp.interpolate(method='linear', limit_direction='forward', inplace=True)
        
        # 恢复尾部原始NaN值
        if last_valid_idx and last_valid_idx < len(ts) - 1:
            ts_temp[tail_mask] = np.nan
        
        return ts_temp

    def _fill_tail_missing(self, ts, time_series, col_name):
        """根据列类型使用不同的尾部填充策略"""
        last_valid_idx = ts.last_valid_index()
        
        # 检查是否存在尾部缺失
        if last_valid_idx is None or last_valid_idx >= len(ts) - 1:
            return ts
            
        # 获取尾部缺失段
        tail_start = last_valid_idx + 1
        tail_na = ts.loc[tail_start:]
        
        # 获取用于拟合的数据点
        fit_data = ts.loc[:last_valid_idx].dropna()
        
        if len(fit_data) == 0:
            # 没有有效点，使用后备填充
            if col_name in self.column_stats:
                return ts.fillna(self.column_stats[col_name]['median'])
            return ts.fillna(0)
        
        # 判断列类型并选择填充策略
        if col_name in self.economic_columns:
            # 经济数据列 - 使用时间序列回归填充
            return self._economic_tail_fill(ts, time_series, fit_data, tail_na)
        else:
            # 非经济数据列 - 使用最后n个有效值均值填充
            # return self._non_economic_tail_fill(ts, time_series, fit_data, tail_na, col_name)
            return self._advanced_tail_forecast(ts, time_series, fit_data, tail_na, col_name)
    
    def _non_economic_tail_fill(self, ts, time_series, fit_data, tail_na, col_name):
        """非经济数据列的尾部填充 - 使用时间序列预测模型"""
        num_missing = len(tail_na)
        # print(num_missing)
        # print(len(tail_na))
        
        # 限制预测步长不超过设定的最大范围
        forecast_steps = min(num_missing, self.forecast_horizon)
        
        # 根据选择的模型进行预测
        try:
            if self.non_economic_model == 'ses':
                # print(forecast_steps)
                # print(fit_data.values)
                # 简单指数平滑 (Simple Exponential Smoothing)
                # print(len(fit_data))
                # print(111)
                # print(fit_data.values[-1])
                model = SimpleExpSmoothing(fit_data.values)
                model_fit = model.fit()
                predicted = model_fit.forecast(forecast_steps)
                # print(predicted)
            elif self.non_economic_model == 'linear':
                # 时间索引线性回归
                x = np.arange(len(fit_data)).reshape(-1, 1)
                y = fit_data.values
                model = LinearRegression()
                model.fit(x, y)
                # 预测未来值
                x_future = np.arange(len(fit_data), len(fit_data) + forecast_steps).reshape(-1, 1)
                predicted = model.predict(x_future)[-forecast_steps:]
            elif self.non_economic_model == 'lastn':
                # 最后n个值的加权平均（近期值权重更高）
                n = min(6, len(fit_data))
                weights = np.linspace(0.1, 1.0, n)  # 线性权重
                weights /= weights.sum()  # 归一化
                last_values = fit_data.iloc[-n:]
                predicted = [np.average(last_values, weights=weights)] * forecast_steps
            else:
                # 默认使用简单指数平滑
                model = SimpleExpSmoothing(fit_data.values)
                model_fit = model.fit()
                predicted = model_fit.forecast(forecast_steps)
            
            # 应用预测结果到尾部缺失段
            if forecast_steps < num_missing:
                # 如果实际缺失超过预测范围，用最后一个预测值填充剩余部分
                ts.loc[tail_na.index[:forecast_steps]] = predicted
                ts.loc[tail_na.index[forecast_steps:]] = predicted[-1]
            else:
                # print(predicted)
                ts.loc[tail_na.index] = predicted
        except Exception as e:
            # 模型失败时使用后备策略
            warnings.warn(f"TS model failed for {col_name}: {str(e)}. Using fallback method.")
            return self._fallback_tail_fill(ts, fit_data, tail_na)
        
        return ts

    def _economic_tail_fill(self, ts, time_series, fit_data, tail_na):
        """经济数据列的尾部填充 - 时间序列回归"""
        # 时间戳处理
        time_values = time_series.astype('int64')
        
        if len(fit_data) < self.min_fit_points:
            # 有效点不足，使用简单外推
            last_value = fit_data.iloc[-1]
            ts.loc[tail_na.index] = last_value
            return ts
        
        try:
            # 时间序列回归
            # print('regression')
            x_fit = time_values.loc[fit_data.index].values.astype(float)
            y_fit = fit_data.values.astype(float)
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
            
            # 预测尾部
            x_predict = time_values.loc[tail_na.index].values.astype(float)
            predicted = slope * x_predict + intercept
            
            # 应用预测结果
            ts.loc[tail_na.index] = predicted
        except Exception as e:
            # 回归失败时使用最后一个有效值
            warnings.warn(f"Regression failed for {col_name}: {str(e)}. Using last valid value.")
            last_value = fit_data.iloc[-1]
            ts.loc[tail_na.index] = last_value
        
        return ts
    
    def _advanced_tail_forecast(self, ts, time_series, fit_data, tail_na, col_name):
        """使用高级时间序列模型预测尾部缺失值"""
        num_missing = len(tail_na)
        forecast_steps = min(num_missing, self.forecast_horizon)
        
        try:
            # ARIMA模型
            if self.non_economic_model == 'arima':
                order = self.model_params.get('arima', {}).get('order', (1,1,1))
                model = ARIMA(fit_data.values, order=order)
                model_fit = model.fit()
                predicted = model_fit.forecast(steps=forecast_steps)
            
            # SARIMA模型（季节性ARIMA）
            elif self.non_economic_model == 'sarima':
                order = self.model_params.get('sarima', {}).get('order', (1,1,1))
                seasonal_order = self.model_params.get('sarima', {}).get('seasonal_order', (1,1,1,12))
                model = SARIMAX(fit_data.values, 
                              order=order, 
                              seasonal_order=seasonal_order)
                model_fit = model.fit(disp=False)
                predicted = model_fit.forecast(steps=forecast_steps)
            
            # Holt-Winters三参数指数平滑
            elif self.non_economic_model == 'holt':
                model = Holt(fit_data.values)
                model_fit = model.fit()
                predicted = model_fit.forecast(steps=forecast_steps)
            
            # 简单指数平滑
            elif self.non_economic_model == 'ses':
                model = SimpleExpSmoothing(fit_data.values)
                model_fit = model.fit()
                predicted = model_fit.forecast(forecast_steps)
            
            # 最后n个值加权平均（简单后备）
            elif self.non_economic_model == 'lastn':
                n = min(6, len(fit_data))
                weights = np.linspace(0.1, 1.0, n)
                weights /= weights.sum()
                last_values = fit_data.iloc[-n:]
                predicted = [np.average(last_values, weights=weights)] * forecast_steps
            
            # 应用预测结果
            if forecast_steps < num_missing:
                ts.loc[tail_na.index[:forecast_steps]] = predicted
                ts.loc[tail_na.index[forecast_steps:]] = predicted[-1]
            else:
                ts.loc[tail_na.index] = predicted
                
        except Exception as e:
            # 模型失败时使用后备策略
            warnings.warn(f"TS model failed for {col_name}: {str(e)}. Using fallback method.")
            ts = self._fallback_tail_fill(ts, fit_data, tail_na)
        
        return ts
    
    def _fallback_tail_fill(self, ts, fit_data, tail_na):
        """后备尾部填充策略"""
        if len(fit_data) > 0:
            # 使用最后3个有效值的加权平均（近期权重更高）
            n = min(3, len(fit_data))
            weights = np.linspace(0.5, 1, n)
            weights /= weights.sum()
            last_values = fit_data.iloc[-n:]
            fill_value = np.average(last_values, weights=weights)
        else:
            # 无有效数据时使用列统计
            fill_value = self.column_stats.get(col_name, {}).get('median', 0)
        
        ts.loc[tail_na.index] = fill_value
        return ts
    
    def _regression_fill(self, ts, time_series):
        """使用时间特征进行回归填充（非尾部）"""
        # 获取有效点
        valid_idx = ts.dropna().index
        if len(valid_idx) < self.min_fit_points:
            return ts.interpolate(method='linear', limit_direction='forward')
        
        # 时间戳处理
        time_values = time_series.astype('int64')
        
        # 准备回归数据
        x = time_values.loc[valid_idx].values.astype(float)
        y = ts.loc[valid_idx].values.astype(float)
        
        try:
            # 线性回归
            slope, intercept, *_ = stats.linregress(x, y)
            
            # 填充非尾部缺失
            na_idx = ts[ts.isna()].index
            if len(na_idx) > 0:
                # 排除尾部缺失（将在后续步骤处理）
                last_valid_idx = ts.last_valid_index()
                if last_valid_idx:
                    na_idx = na_idx[na_idx <= last_valid_idx]
                
                if len(na_idx) > 0:
                    x_na = time_values.loc[na_idx].values.astype(float)
                    ts.loc[na_idx] = slope * x_na + intercept
        except:
            # 回归失败时使用线性插值
            ts = ts.interpolate(method='linear', limit_direction='forward')
        
        return ts



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
        self.feature_builder = feature_builder or FeatureBuilder(TimeGranularityController(granularity))
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
        # data.to_csv(f'./result/{origin}_{destination}_all_data.csv', index=False)

        # 调整测试集大小（按粒度转换）
        if self.granularity_controller.granularity == 'quarterly':
            test_size = min(2*3, test_size)  # 至少2个季度 还是按照月度走
        elif self.granularity_controller.granularity == 'yearly':
            # test_size = max(1, test_size // 12)  # 至少1年
            test_size = 0  # 不使用测试集
        
        # 数据预处理
        data_preprocessed = self.preprocessor.fit_transform(data)
        # data_preprocessed.to_csv(f'./result/{origin}_{destination}_data_preprocessed.csv', index=False)
        
        # 特征工程 - 先fit再transform
        self.feature_builder.fit(data_preprocessed)
        data_with_features = self.feature_builder.transform(data_preprocessed)
        # data_with_features.to_csv(f'./result/{origin}_{destination}_data_with_features.csv', index=False)
        
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



# class DataPreprocessor(BaseEstimator, TransformerMixin):
#     """高级数据预处理类，解决尾部缺失检测和填充问题"""
#     def __init__(self, fill_method='interp', max_invalid_ratio=1, min_fit_points=7, 
#                  normalize=False, time_col='YearMonth', tail_fill_method='extrapolate'):
#         """
#         Args:
#             fill_method (str): 主填充方法 ['interp'|'zero'|'regression']
#             max_invalid_ratio (float): 最大允许缺失值比例
#             min_fit_points (int): 回归填充所需的最小有效点数
#             normalize (bool): 是否执行归一化
#             time_col (str): 时间列名称
#             tail_fill_method (str): 尾部填充专用方法 ['regression'|'extrapolate']
#         """
#         self.fill_method = fill_method
#         self.max_invalid_ratio = max_invalid_ratio
#         self.min_fit_points = min_fit_points
#         self.normalize = normalize
#         self.time_col = time_col
#         self.tail_fill_method = tail_fill_method
#         self.scaler = None
#         self.binary_columns = []
#         self.time_index_map = {}
#         self.column_stats = {}
        
#     def fit(self, X, y=None):
#         # 识别0-1列
#         self.binary_columns = [
#             col for col in X.columns 
#             if col != self.time_col and set(X[col].dropna().unique()).issubset({0, 1})
#         ]
        
#         # 创建全局时间索引映射
#         self.time_index_map = pd.to_datetime(X[self.time_col]).astype('int64').to_dict()
        
#         # 收集列统计信息用于后备填充
#         for col in X.columns:
#             if col != self.time_col and col not in self.binary_columns:
#                 non_zero = X[col][(X[col] != 0) & (X[col].notna())]
#                 if len(non_zero) > 0:
#                     self.column_stats[col] = {
#                         'mean': non_zero.mean(),
#                         'median': non_zero.median(),
#                         'min': non_zero.min(),
#                         'max': non_zero.max()
#                     }
#         return self
        
#     def transform(self, X):
#         # 深拷贝数据并确保时间列存在
#         if self.time_col not in X.columns:
#             raise ValueError(f"Time column '{self.time_col}' not found in data")
            
#         X_filled = X.copy()
#         time_series = pd.to_datetime(X_filled[self.time_col])
        
#         # 第一步：将0值转换为NaN（0-1列除外）
#         for col in X.columns:
#             if col != self.time_col and col not in self.binary_columns:
#                 X_filled.loc[X_filled[col] == 0, col] = np.nan
        
#         # 第二步：按列进行填充处理
#         for col in X.columns:
#             if col in self.binary_columns or col == self.time_col:
#                 continue
                
#             ts = X_filled[col].copy()
#             original_na_count = ts.isna().sum()
            
#             if ts.isna().mean() > self.max_invalid_ratio:
#                 # 缺失过多，使用后备填充
#                 if col in self.column_stats:
#                     ts.fillna(self.column_stats[col]['median'], inplace=True)
#                 else:
#                     ts.fillna(0, inplace=True)
#                 X_filled[col] = ts
#                 continue
                
#             # 处理头部缺失
#             first_valid_idx = ts.first_valid_index()
#             if first_valid_idx is not None:
#                 # 使用第一个有效值向前填充头部
#                 first_valid_value = ts.loc[first_valid_idx]
#                 ts.loc[:first_valid_idx] = first_valid_value
#             else:
#                 # 无有效值，使用列统计
#                 if col in self.column_stats:
#                     ts.fillna(self.column_stats[col]['median'], inplace=True)
#                 else:
#                     ts.fillna(0, inplace=True)
#                 X_filled[col] = ts
#                 continue
                
#             # 主填充方法
#             if self.fill_method == 'interp':
#                 # 仅填充中间缺失，保留尾部缺失
#                 ts = self._safe_interpolate(ts, time_series)
#             elif self.fill_method == 'regression':
#                 ts = self._regression_fill(ts, time_series)
#             elif self.fill_method == 'zero':
#                 ts.fillna(0, inplace=True)
            
#             # 专门处理尾部缺失
#             ts = self._fill_tail_missing(ts, time_series)
                
#             # 最终后备填充（处理任何剩余缺失）
#             if ts.isna().any():
#                 if col in self.column_stats:
#                     ts.fillna(self.column_stats[col]['median'], inplace=True)
#                 else:
#                     # 线性插值作为最后手段
#                     ts.interpolate(method='linear', limit_direction='both', inplace=True)
            
#             X_filled[col] = ts
        
#         # 0-1列特殊处理
#         for col in self.binary_columns:
#             if X_filled[col].isna().any():
#                 mode_val = X_filled[col].mode()
#                 if not mode_val.empty:
#                     X_filled.loc[:, col] = X_filled[col].fillna(mode_val[0])
#                 else:
#                     X_filled.loc[:, col] = X_filled[col].fillna(0)
        
#         # 归一化处理
#         if self.normalize:
#             non_binary_cols = [c for c in X.columns if c not in self.binary_columns and c != self.time_col]
#             if non_binary_cols:
#                 self.scaler = StandardScaler()
#                 X_filled[non_binary_cols] = self.scaler.fit_transform(X_filled[non_binary_cols])
            
#         return X_filled

#     def _safe_interpolate(self, ts, time_series):
#         """安全插值，避免填充尾部"""
#         # 创建临时副本用于插值
#         ts_temp = ts.copy()
        
#         # 标记尾部缺失位置
#         last_valid_idx = ts_temp.last_valid_index()
#         if last_valid_idx and last_valid_idx < len(ts_temp) - 1:
#             tail_mask = ts_temp.index > last_valid_idx
#             ts_temp[tail_mask] = np.nan  # 确保尾部保持为NaN
        
#         # 执行插值（仅限非尾部）
#         ts_temp.interpolate(method='linear', limit_direction='forward', inplace=True)
        
#         # 恢复尾部原始NaN值
#         if last_valid_idx and last_valid_idx < len(ts) - 1:
#             ts_temp[tail_mask] = np.nan
        
#         return ts_temp

#     def _fill_tail_missing(self, ts, time_series):
#         """专门处理尾部缺失值"""
#         last_valid_idx = ts.last_valid_index()
        
#         # 检查是否存在尾部缺失
#         if last_valid_idx is None or last_valid_idx >= len(ts) - 1:
#             return ts
            
#         # 获取尾部缺失段
#         tail_start = last_valid_idx + 1
#         tail_na = ts.loc[tail_start:]
        
#         # 获取用于拟合的数据点
#         fit_data = ts.loc[:last_valid_idx].dropna()
        
#         if len(fit_data) < self.min_fit_points:
#             # 有效点不足，使用简单外推
#             last_value = fit_data.iloc[-1] if len(fit_data) > 0 else 0
#             ts.loc[tail_start:] = last_value
#             return ts
        
#         # 时间戳处理
#         time_values = time_series.astype('int64')
        
#         if self.tail_fill_method == 'regression':
#             # 时间序列回归
#             x_fit = time_values.loc[fit_data.index].values.astype(float)
#             y_fit = fit_data.values.astype(float)
            
#             # 线性回归
#             slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
            
#             # 预测尾部
#             x_predict = time_values.loc[tail_na.index].values.astype(float)
#             predicted = slope * x_predict + intercept
            
#             # 应用预测结果
#             ts.loc[tail_na.index] = predicted
            
#         elif self.tail_fill_method == 'extrapolate':
#             # 使用最后N个点进行外推
#             recent_points = fit_data.iloc[-self.min_fit_points:]
#             x_fit = np.arange(len(recent_points))
#             y_fit = recent_points.values.astype(float)
            
#             # 线性外推
#             slope, intercept, *_ = stats.linregress(x_fit, y_fit)
            
#             # 预测尾部
#             for i, idx in enumerate(tail_na.index):
#                 ts.loc[idx] = slope * (len(recent_points) + i) + intercept
        
#         return ts

#     def _regression_fill(self, ts, time_series):
#         """使用时间特征进行回归填充（非尾部）"""
#         # 获取有效点
#         valid_idx = ts.dropna().index
#         if len(valid_idx) < self.min_fit_points:
#             return ts.interpolate(method='linear', limit_direction='forward')
        
#         # 时间戳处理
#         time_values = time_series.astype('int64')
        
#         # 准备回归数据
#         x = time_values.loc[valid_idx].values.astype(float)
#         y = ts.loc[valid_idx].values.astype(float)
        
#         # 线性回归
#         slope, intercept, *_ = stats.linregress(x, y)
        
#         # 填充非尾部缺失
#         na_idx = ts[ts.isna()].index
#         if len(na_idx) > 0:
#             # 排除尾部缺失（将在后续步骤处理）
#             last_valid_idx = ts.last_valid_index()
#             if last_valid_idx:
#                 na_idx = na_idx[na_idx <= last_valid_idx]
            
#             if len(na_idx) > 0:
#                 x_na = time_values.loc[na_idx].values.astype(float)
#                 ts.loc[na_idx] = slope * x_na + intercept
        
#         return ts


# class DataPreprocessor(BaseEstimator, TransformerMixin):
#     """数据预处理类，处理缺失值填充和归一化"""
#     def __init__(self, fill_method='interp', max_invalid_ratio=1, min_fit_points=3, normalize=False):
#         """
#         Args:
#             fill_method (str): 填充方法 ['interp'|'zero'|'regression']
#             max_invalid_ratio (float): 最大允许缺失值比例
#             min_fit_points (int): 回归填充所需的最小有效点数
#             normalize (bool): 是否执行归一化
#         """
#         self.fill_method = fill_method
#         self.max_invalid_ratio = max_invalid_ratio
#         self.min_fit_points = min_fit_points
#         self.normalize = normalize
#         self.scaler = None
#         self.binary_columns = []
        
#     def fit(self, X, y=None):
#         # 识别0-1列
#         self.binary_columns = [
#             col for col in X.columns 
#             if set(X[col].dropna().unique()).issubset({0, 1})
#         ]
#         return self
        
#     def transform(self, X):
#         X_filled = X.copy()
        
#         # 填充处理
#         for col in X.columns:
#             if col in self.binary_columns:
#                 continue  # 跳过0-1列
                
#             ts = X[col].copy()
#             invalid_mask = ts.isna() | (ts == 0)
#             invalid_ratio = invalid_mask.mean()
            
#             if invalid_ratio > self.max_invalid_ratio:
#                 continue
                
#             ts[ts == 0] = np.nan
                
#             if self.fill_method == 'interp':
#                 ts_filled = ts.interpolate(method='linear', limit_direction='both')
#                 ts_filled = self._handle_tail_missing(ts_filled)
                
#             elif self.fill_method == 'regression':
#                 ts_filled = self._regression_fill(ts)
                
#             elif self.fill_method == 'zero':
#                 ts_filled = ts.fillna(0)
                
#             X_filled[col] = ts_filled
        
#         # 归一化处理
#         if self.normalize:
#             non_binary_cols = [c for c in X.columns if c not in self.binary_columns]
#             self.scaler = StandardScaler()
#             X_filled[non_binary_cols] = self.scaler.fit_transform(X_filled[non_binary_cols])
            
#         return X_filled

#     def _handle_tail_missing(self, ts):
#         """处理尾部缺失值"""
#         if ts.isna().any():
#             last_valid_idx = ts.last_valid_index()
#             if last_valid_idx is not None and last_valid_idx < len(ts) - 1:
#                 recent_valid = []
#                 idx = last_valid_idx
#                 while idx >= 0 and not pd.isna(ts.iloc[idx]):
#                     recent_valid.append((idx, ts.iloc[idx]))
#                     idx -= 1
                
#                 if len(recent_valid) >= self.min_fit_points:
#                     recent_valid = recent_valid[::-1]
#                     x = np.arange(len(recent_valid))
#                     y = np.array([v for _, v in recent_valid])
#                     slope, intercept, *_ = stats.linregress(x, y)
                    
#                     for i in range(last_valid_idx + 1, len(ts)):
#                         ts.iloc[i] = slope * (i - recent_valid[0][0]) + intercept
#         return ts

#     def _regression_fill(self, ts):
#         """回归填充整个序列"""
#         # 实现略，根据实际特征工程需要调整
#         return ts.interpolate(method='linear', limit_direction='both')
