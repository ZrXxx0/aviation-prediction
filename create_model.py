import xgboost as xgb
import lightgbm as lgb



def get_model(granularity, model_type='lgb'):
    """根据时间粒度和模型类型返回配置好的模型"""
    if model_type == 'lgb':
        if granularity == 'monthly':
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=7,
                min_child_samples=10,
                min_split_gain=0.0,
                learning_rate=0.1,
                random_state=42
            )
        elif granularity == 'quarterly':
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=2,
                min_child_samples=2,
                min_split_gain=0.0,
                learning_rate=0.1,
                random_state=42
            )
        else:  # yearly
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                min_child_samples=1,
                min_split_gain=0.0,
                learning_rate=0.1,
                random_state=42
            )
    else:  # xgb
        if granularity == 'monthly':
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                reg_lambda=1,
                random_state=42
            )
        else:  # quarterly/yearly
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                reg_lambda=1,
                random_state=42
            )