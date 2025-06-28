import os
import pandas as pd
import numpy as np

from FeatureEngineer import AirlineRouteModel

# 检查数据是否有效（非空且非零）
def is_valid(x):
    if pd.isna(x) or x == 0 or x == '0' or (isinstance(x, str) and x.strip() == '0'):
        return False
    return True

# 读取CSV文件
domestic = pd.read_csv('./final_data_0622.csv', low_memory=False)

# 获取所有唯一的航线对
unique_routes = domestic[['Origin', 'Destination']].drop_duplicates()

# 初始化结果列表
results = []

# 初始化路由处理器
route_processor = AirlineRouteModel(data=domestic)

# 遍历每条唯一航线
for idx, row in unique_routes.iterrows():
    origin = row['Origin']
    destination = row['Destination']
    
    try:
        # 获取航线数据
        route_data = route_processor.get_route_data(origin, destination)
        
        # 计算统计量
        total_cells = route_data.size  # 总数据单元格数
        if total_cells == 0:
            valid_count = 0
            ratio = 0.0
        else:
            # 计算有效数据数量
            valid_mask = route_data.map(is_valid)
            valid_count = valid_mask.sum().sum()
            ratio = valid_count / total_cells
        
        # 添加结果
        results.append({
            'Origin': origin,
            'Destination': destination,
            'Total_Cells': total_cells,
            'Valid_Cells': valid_count,
            'Valid_Ratio': ratio
        })
        
    except Exception as e:
        print(f"Error processing {origin}-{destination}: {str(e)}")
        results.append({
            'Origin': origin,
            'Destination': destination,
            'Total_Cells': 0,
            'Valid_Cells': 0,
            'Valid_Ratio': 0.0
        })

# 创建结果DataFrame并排序
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Valid_Ratio', ascending=False)

# 保存到CSV文件
results_df.to_csv('./route_data_report.csv', index=False)

print("处理完成! 结果已保存到 route_data_report.csv")