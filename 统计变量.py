import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('./final_data_0614.csv', low_memory=False)


# # 统计确定距离分箱
# distance = df['Distance (KM)']

# print('基本统计量:\n', distance.describe())
# print('\n百分位数:', np.percentile(distance.dropna(), [10, 25, 50, 75, 90]))  # [ 587.  887. 1324. 1967. 3428.]

# # 可视化分布
# plt.figure(figsize=(10,6))
# sns.histplot(distance, bins=30, kde=True)
# plt.title('Distance Distribution')
# plt.show()

# 确定机型
equipment_counts = df['Equipment'].value_counts(dropna=False)
print('\n机型分布:\n', equipment_counts)

# 可视化机型分布（可选）
plt.figure(figsize=(12,6))
sns.barplot(x=equipment_counts.index, y=equipment_counts.values)
plt.title('Equipment Distribution')
plt.xticks(rotation=45)
plt.show()

# 无法仅凭机型就确定一个精确的座位数 所以这个目前就不加了！