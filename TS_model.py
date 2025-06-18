import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

class BaseTSModel:
    """基础时间序列模型接口"""
    def fit(self, series):
        """训练时间序列模型"""
        pass
    
    def predict(self, dates):
        """预测指定日期序列"""
        pass

class ARIMAModel(BaseTSModel):
    """ARIMA 时间序列模型实现"""
    def __init__(self, order=(1, 0, 0), freq=None):
        self.order = order
        self.model = None
        self.last_training_date = None
        self.freq = freq  # 频率参数
        
    def fit(self, series):
        # 确保有足够的数据点
        if len(series) < max(self.order) * 2:
            return  # 数据不足，跳过训练
            
        try:
            # print(series)
            # if not hasattr(series.index, 'freq') or series.index.freq is None:
            #     freq = self.freq if self.freq else 'MS'  # 默认为月
            #     series = series.asfreq(freq)
            # print(self.order)
            self.model = ARIMA(series, order=self.order).fit()
            self.last_training_date = series.index[-1]
        except Exception as e:
            print(f"ARIMA 训练失败: {e}")
            self.model = None

    def predict(self, dates):
        if self.model is None:
            return pd.Series(index=dates, dtype=float)
            
        # 创建完整时间索引的序列
        freq = self.freq if self.freq else'MS'
        # print(freq)
        full_index = pd.date_range(
            start=dates.min(), 
            end=dates.max(), 
            freq='D'
        )
        # print(full_index)
        full_series = pd.Series(index=full_index, dtype=float)
        
        # 填充历史拟合值
        if hasattr(self.model, 'fittedvalues'):
            fitted_values = self.model.fittedvalues
            full_series.loc[fitted_values.index] = fitted_values
        # print(full_series)
        # 预测未来值
        if dates.max() > self.last_training_date:
            future_periods = len(dates[dates > self.last_training_date])
            forecast = self.model.forecast(steps=future_periods)
            if freq =='MS':
                offset = pd.DateOffset(months=1)
            elif freq == 'QS':
                offset = pd.DateOffset(months=3)
            else:  # YS
                offset = pd.DateOffset(years=1)
            forecast_index = pd.date_range(
                start=self.last_training_date + offset,
                periods=future_periods,
                freq='D'
            )
            # print(full_series)
            # 确保索引包含预测日期
            missing_dates = forecast_index.difference(full_series.index)
            if not missing_dates.empty:
                full_series = full_series.reindex(full_series.index.union(missing_dates))
            
            # 进行预测
            forecast = self.model.forecast(steps=future_periods)
            full_series.loc[forecast_index] = forecast.values
            
        # 对齐到请求的日期
        # print(full_series)
        return full_series.reindex(dates)

class ProphetModel(BaseTSModel):
    """Prophet 时间序列模型实现"""
    def __init__(self, yearly_seasonality=True):
        self.model = None
        self.yearly_seasonality = yearly_seasonality
        
    def fit(self, series):
        if len(series) < 12:  # Prophet 需要至少1年数据
            return
            
        try:
            # 准备 Prophet 格式的数据
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            self.model = Prophet(yearly_seasonality=self.yearly_seasonality)
            self.model.fit(df)
        except Exception as e:
            print(f"Prophet 训练失败: {e}")
            self.model = None

    def predict(self, dates):
        if self.model is None:
            return pd.Series(index=dates, dtype=float)
            
        # 创建预测数据框
        future = pd.DataFrame({'ds': dates})
        forecast = self.model.predict(future)
        return pd.Series(forecast['yhat'].values, index=dates)