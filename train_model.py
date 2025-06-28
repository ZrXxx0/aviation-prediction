import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

from time_granularity import TimeGranularityController
from TS_model import ARIMAModel, ProphetModel
from model_evaluation import ModelEvaluator
from FeatureEngineer import DataPreprocessor, FeatureBuilder, AirlineRouteModel


##################################    控制数据   ##################################
# 航线选择
origin='PEK'
destination='SZX'
domestic = pd.read_csv('./final_data_0622.csv', low_memory=False)

# 测试集大小
test_size=12  # 最后12个月/4季度/1年为测试集
# test_size=6  # 季度的时候，用两个季度作为测试集会好一些，就是6个月

# 训练时间粒度
TIME_GRANULARITY = 'quarterly'  # 'monthly', 'quarterly', 'yearly'

# 是否增加时间序列预测特征
add_ts_forecast=True  # True or False
# 目前支持ARIMA 模型, Prophet 模型需要进一步调试，其他时间序列模型可以补充在ts_model.py文件中

# 未来预测时长
future_periods = 36  # 默认36个月/12季度/4年 以月为单位



##################################    模型训练   ##################################
# 创建处理实例
preprocessor = DataPreprocessor(
    fill_method='interp', 
    normalize=False,
    non_economic_tail_window=6,  # 非经济指标的滑动窗口大小
)

# 初始化粒度控制器
granularity_controller = TimeGranularityController(TIME_GRANULARITY)
# print(granularity_controller.get_freq())  

# 创建时间序列模型 - 选择其中一种
ts_model = ARIMAModel(
    order=(1,1,1), 
    freq=granularity_controller.get_freq()  # 设置频率
)  # ARIMA 模型
# ts_model = ProphetModel(yearly_seasonality=True)  # Prophet 模型

feature_builder = FeatureBuilder(
    granularity_controller=granularity_controller,  # 传入粒度控制器
    add_ts_forecast=add_ts_forecast,  # 启用时间序列特征
    ts_model=ts_model  # 传入时间序列模型
)

# 创建航线处理器
route_processor = AirlineRouteModel(
    data=domestic,
    preprocessor=preprocessor,
    feature_builder=feature_builder,
    granularity=TIME_GRANULARITY
)

# 获取特定航线的训练/测试数据
X_train, y_train, X_test, y_test, data_with_features = route_processor.prepare_data(
    origin=origin,
    destination=destination,
    test_size=test_size
)

save_dir = f'./result/{TIME_GRANULARITY}'
os.makedirs(save_dir, exist_ok=True)
X_train.to_csv(f'{save_dir}/{origin}_{destination}_X_train.csv', index=False)
X_test.to_csv(f'{save_dir}/{origin}_{destination}_X_test.csv', index=False)
pd.DataFrame(y_train).to_csv(f'{save_dir}/{origin}_{destination}_y_train.csv', index=False)
pd.DataFrame(y_test).to_csv(f'{save_dir}/{origin}_{destination}_y_test.csv', index=False)
data_with_features.to_csv(f'{save_dir}/{origin}_{destination}_data_with_features.csv', index=False)
# print(X_train.shape)

"""
模型选择如下
"""
# 模型加载 monthly
model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=7,
    min_child_samples=10,         # 更小的叶子节点允许更多分裂
    min_split_gain=0.0,           # 放宽分裂的最小增益门槛
    learning_rate=0.1,
    random_state=42
)
# quarterly
model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=2,
    min_child_samples=2,         # 更小的叶子节点允许更多分裂
    min_split_gain=0.0,           # 放宽分裂的最小增益门槛
    learning_rate=0.1,
    random_state=42
)
# yearly
# model = lgb.LGBMRegressor(
#     n_estimators=100,
#     max_depth=3,
#     min_child_samples=1,         # 更小的叶子节点允许更多分裂
#     min_split_gain=0.0,           # 放宽分裂的最小增益门槛
#     learning_rate=0.1,
#     random_state=42
# )
# 模型加载 monthly
# model = xgb.XGBRegressor(
#     n_estimators=100,
#     max_depth=3,
#     learning_rate=0.1,
#     reg_lambda=1,  
#     random_state=42
# )
# yearly quarterly
# model = xgb.XGBRegressor(
#     n_estimators=100,
#     max_depth=2,
#     learning_rate=0.1,
#     reg_lambda=1,  
#     random_state=42
# )
# 训练模型
model.fit(X_train, y_train)



##################################    模型评估   ##################################
# 特征重要性
importances = model.feature_importances_
lgb.plot_importance(model, max_num_features=20)
# xgb.plot_importance(model, max_num_features=20)
plt.show()

# 评估模型
train_preds = model.predict(X_train)
train_evaluator = ModelEvaluator(y_train, train_preds).calculate_metrics()
train_evaluator.report("Train")
test_preds = None
if TIME_GRANULARITY != 'yearly':
    test_preds = model.predict(X_test)
    test_evaluator = ModelEvaluator(y_test, test_preds).calculate_metrics()
    test_evaluator.report("Test")



##################################    模型预测   ##################################
# 测试集预测
history = data_with_features.copy()
feature_cols = X_train.columns.tolist()
date_col = route_processor.date_col
target_col = route_processor.target_col

# 未来预测

if TIME_GRANULARITY == 'quarterly':
    future_periods = future_periods//3  # 预测12个季度
elif TIME_GRANULARITY == 'yearly':
    future_periods = future_periods//12  # 预测3年

latest_data = data_with_features.copy()
last_complete_date = data_with_features[date_col].max()
if TIME_GRANULARITY == 'quarterly':
    # 季度数据：确保起始点是季度末（3、6、9、12月）
    while last_complete_date.month not in [3, 6, 9, 12]:
        last_complete_date -= pd.DateOffset(months=1)
elif TIME_GRANULARITY == 'yearly':
    # 年度数据：确保起始点是年末（12月）
    while last_complete_date.month != 12:
        last_complete_date -= pd.DateOffset(months=1)
future_preds = []

for i in range(future_periods):
    # 根据粒度调整日期增量
    if TIME_GRANULARITY == 'monthly':
        offset = pd.DateOffset(months=1)
    elif TIME_GRANULARITY == 'quarterly':
        offset = pd.DateOffset(months=3)
    else:  # yearly
        offset = pd.DateOffset(years=1)
    next_date = last_complete_date + offset
    next_row = {'YearMonth': next_date}
    latest_data = pd.concat([latest_data, pd.DataFrame([next_row])], ignore_index=True)
    latest_data = route_processor.preprocessor.fit_transform(latest_data)
    latest_data = route_processor.feature_builder.fit_transform(latest_data)
    latest_input = latest_data.iloc[[-1]][feature_cols]
    next_pred = model.predict(latest_input)[0]
    latest_data.loc[latest_data.index[-1], 'Route_Total_Seats'] = next_pred
    future_preds.append({'YearMonth': next_date, 'Predicted': next_pred})
    last_complete_date = next_date

latest_data.to_csv(f'{save_dir}/{origin}_{destination}_latest_data.csv', index=False)

# 创建结果DataFrame
train_df = pd.DataFrame({
    'YearMonth': history[date_col].iloc[:len(X_train)],
    'Actual': y_train,
    'Predicted': train_preds,
    'Set': 'Train'
})

test_df = pd.DataFrame({
    'YearMonth': history[date_col].iloc[len(X_train):len(X_train)+len(X_test)],
    'Actual': y_test,
    'Predicted': test_preds,
    'Set': 'Test'
})

future_df = pd.DataFrame(future_preds)
future_df['Actual'] = np.nan
future_df['Set'] = 'Future'

if test_df.empty:  # 如果没有测试集数据，只保留训练集和未来预测
    result_df = pd.concat([train_df, future_df])
else:  # 如果有测试集数据，保留训练集、测试集和未来预测
    result_df = pd.concat([train_df, test_df, future_df])
# result_df.to_csv(f'{save_dir}/{origin}_{destination}_预测结果.csv', 
#                 index=False, 
#                 encoding='utf_8_sig',  # 支持中文路径
#                 date_format='%Y-%m')  # 统一日期格式



##################################    模型可视化   ##################################
# 可视化
plt.figure(figsize=(14, 6))
plt.plot(result_df['YearMonth'], result_df['Actual'], label='Actual', color='black')

# 训练集预测
train_mask = result_df['Set'] == 'Train'
plt.plot(result_df[train_mask]['YearMonth'], 
            result_df[train_mask]['Predicted'], 
            label='Train Pred', linestyle='--', color='blue')

# 测试集预测
if not test_df.empty:  # 如果有测试集数据
    test_mask = result_df['Set'] == 'Test'
    # print(result_df[test_mask])
    plt.plot(result_df[test_mask]['YearMonth'], 
                result_df[test_mask]['Predicted'], 
                label='Test Pred', linestyle='--', color='red', marker='o', markersize=3, )

# 未来预测
future_mask = result_df['Set'] == 'Future'
plt.plot(result_df[future_mask]['YearMonth'], 
            result_df[future_mask]['Predicted'], 
            label='Future Forecast', linestyle='--', color='green')

# 添加分割线
if not test_df.empty:
    split_date = test_df['YearMonth'].min()
    plt.axvline(x=split_date, color='gray', linestyle=':', label='Train/Test Split')

plt.title(f"Route_Total_Seats Forecasting: {origin} to {destination}")
plt.xlabel("Date")
plt.ylabel("Seats")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

