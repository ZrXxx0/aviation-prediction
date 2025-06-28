import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

class DataCleaner:
    def __init__(self, airport_file, economy_file, capacity_input_folder, connection_input_folder, mix_input_folder, final_output_file,
                 capacity_output_folder="./Capacity_Processed_0614",
                 connection_output_folder="./Connection_Processed_0614",
                 mix_output_folder="./Mix_Processed_0614",
                 start_year=2011):
        self.airport_file = airport_file
        self.economy_file = economy_file
        self.capacity_input_folder = capacity_input_folder
        self.capacity_output_folder = capacity_output_folder
        self.connection_input_folder = connection_input_folder
        self.connection_output_folder = connection_output_folder
        self.mix_input_folder = mix_input_folder
        self.mix_output_folder = mix_output_folder
        self.final_output_file = final_output_file
        self.start_year = start_year
        self.OAG_Capacity_columns_to_keep = [
            'Origin', 'Destination', 'Departure Time', 'Arrival Time',
            'Elapsed Time', 'Distance (KM)', 'Equipment', 'Service', 'Stops',
            'Full Itinerary', 'Frequency', 'Seats', 'Time series'
        ]
        self.OAG_Connection_columns = [
            'Origin', 'Destination',
            'Total Est. Pax', 'First', 'Business', 'Premium', 'Full Y', 'Disc Y',
            'Avg yield', 'Avg First', 'Avg Business', 'Avg Premium', 'Avg Full Y', 'Avg Disc Y',
            'International Flights', 'Region'
        ]
        self.OAG_mix_columns = [
            "Market Pair",
            "Total Est. Pax","Local Est. Pax","Behind Est. Pax","Bridge Est. Pax","Beyond Est. Pax",
            "Avg Fare (USD)","Local Fare","Behind Fare","Bridge Fare","Beyond Fare"
        ]
        self.min_origin_destinations = {}
        self.domestic_airports = self.__read_domestic_airports()

    def __read_domestic_airports(self):
        """读取国内机场三字码"""
        try:
            df = pd.read_excel(self.airport_file)
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            return set(df[df['三字码'].notna()]['三字码'].unique())
        except Exception as e:
            print(f"初始化国内机场三字码失败: {str(e)}")
            raise

    def __process_capacity_and_save(self, file_path, output_folder):
        columns_to_keep = self.OAG_Capacity_columns_to_keep
        df = pd.read_csv(file_path)
        df = df[columns_to_keep]
        df = df.drop_duplicates()

        # 处理飞行时间
        df['flighting_time'] = df['Elapsed Time'].str.split(':').apply(
            lambda x: int(x[0])*60 + int(x[1]) if isinstance(x, list) and len(x)==2 else 0
        )
        # 转换日期格式并提取年月
        df['Time series'] = pd.to_datetime(df['Time series'], errors='coerce')
        df = df.dropna(subset=['Time series'])  # 删除无效日期
        df['YearMonth'] = df['Time series'].dt.to_period('M')
        df['Time series'] = df['Time series'].dt.strftime('%Y-%m-%d')

        # 筛选国内航线
        df = df[
            (df['Origin'].isin(self.domestic_airports)) | 
            (df['Destination'].isin(self.domestic_airports))
        ]
        df['International Flights'] = df.apply(
            lambda row: 0 if (row['Origin'] in self.domestic_airports and 
                            row['Destination'] in self.domestic_airports) else 1,
            axis=1
        )
        
        # 记录每个 Origin-Destination 对的最小 Distance
        self.min_origin_destinations.update(df.groupby(['Origin', 'Destination'])['Distance (KM)'].min().to_dict())

        # 按年月分组处理
        for (year_month), group in df.groupby('YearMonth'):
            # 生成目标文件名 (格式: output_folder/YYYY-MM/YYYY-MM.csv)
            ym_str = str(year_month)
            output_file = os.path.join(output_folder, f"cap_{ym_str}.csv")

            # 先计算每个航线的总数（不分机型）
            route_totals = group.groupby(['Time series', 'Origin', 'Destination']).agg(
                Route_Total_Flights=('Seats', 'count'),
                Route_Total_Seats=('Seats', 'sum'),
                Route_Total_Flight_Time=('flighting_time', 'sum'),
                Route_Avg_Flight_Time=('flighting_time', 'mean') 
            ).reset_index()
            
            # 分组统计（按机型）
            result = group.groupby(['Time series','Origin', 'Destination', 'Service', 'Stops', 'Equipment', 'International Flights']).agg(
                Equipment_Total_Flights=('Seats', 'count'),    #航班数量
                Equipment_Total_Seats=('Seats', 'sum'),    #座位数量
                # Equipment_Total_Flight_Time=('flighting_time', 'sum')    #飞行时间
            ).reset_index()
            
            # 合并总数到结果中
            result = pd.merge(result, route_totals, 
                            on=['Time series', 'Origin', 'Destination'],
                            how='left')
            
            result = result.sort_values(by=['Time series', 'Route_Total_Flights', 'Route_Total_Seats', 'Route_Total_Flight_Time', 'Route_Avg_Flight_Time', 'Equipment_Total_Flights', 'Equipment_Total_Seats'], 
                                        ascending=[True, False, False, False, False, False, False])
            
            # 检查文件是否存在
            if os.path.exists(output_file):
                # 如果文件存在，读取现有数据并追加
                existing_df = pd.read_csv(output_file)
                combined_df = pd.concat([existing_df, result])
                combined_df = combined_df.drop_duplicates().sort_values(by=['Time series', 'Route_Total_Flights', 
                                                                            'Route_Total_Seats', 'Route_Total_Flight_Time', 'Route_Avg_Flight_Time', 'Equipment_Total_Flights', 'Equipment_Total_Seats'], 
                                                                        ascending=[True, False, False, False, False, False, False])
                combined_df.to_csv(output_file, index=False)
            else:
                result.to_csv(output_file, index=False)

    def process_capacity(self):
        """处理oag的capacity数据"""
        input_folder = self.capacity_input_folder
        output_folder = self.capacity_output_folder
        if os.path.exists(output_folder):
            print(f"输出目录 {output_folder} 已存在，跳过process_capacity")
            return
        Path(output_folder).mkdir(parents=True, exist_ok=True) 
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.CSV') or file_name.endswith('.csv'):
                if int(file_name.split('_')[0]) < self.start_year:
                    continue
                file_path = os.path.join(input_folder, file_name)
                print(f"Processing capacity file: {file_path}")
                self.__process_capacity_and_save(file_path, output_folder)
        # 将 min_distances 转换为 DataFrame
        min_distances_df = pd.DataFrame(list(self.min_origin_destinations.items()), columns=['Origin-Destination', 'Distance (KM)'])

        # 将 Origin-Destination 列格式化为字符串格式（如 AAT-AKU）
        min_distances_df['Origin-Destination'] = min_distances_df['Origin-Destination'].apply(lambda x: f"{x[0]}-{x[1]}")

        # 读取所有处理后的文件并拼接最小距离
        for file_name in os.listdir(output_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(output_folder, file_name)
                # print(f"Updating file: {file_path}")
                df = pd.read_csv(file_path)
                df['Origin-Destination'] = df['Origin'] + '-' + df['Destination']
                df = pd.merge(df, min_distances_df, on='Origin-Destination', how='left')
                df.drop(columns=['Origin-Destination'], inplace=True)  # 删除辅助列
                df.to_csv(file_path, index=False)
    
    def __process_connection_and_save(self, file_path, output_folder):
        try:
            # 解析文件名以提取年份和月份
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            if len(parts) < 3:
                print(f"警告：文件名 {filename} 格式不正确。")
                return
            
            region = parts[1]  # 地区
            yyyymm = parts[2].split('_')[0]  # 年月
            yyyy, mm = int(yyyymm[:4]), int(yyyymm[4:])
            yyyymm_str = f"{yyyy:04d}-{mm:02d}"  # 格式化为 YYYY-MM

            # 读取数据
            df = pd.read_csv(file_path, index_col=False)
            
            # 筛选从 AM 到 BB 的列（索引范围 38 到 54）
            columns_to_keep = df.columns[38:55]  # 包括 38 和 54
            df = df[columns_to_keep]
            
            # 删除 Origin 列为空的行
            df = df.dropna(subset=['Origin'])
            
            # 重新命名列名
            new_column_names = [
                'Origin', 'Connecting over', 'Connecting over (Name)', 'Destination',
                'Total Est. Pax', 'First', 'Business', 'Premium', 'Full Y', 'Disc Y',
                'Avg_yield', 'Avg First', 'Avg Business', 'Avg Premium', 'Avg Full Y', 'Avg Disc Y'
            ]
            df.columns = new_column_names
            
            # 筛选国内机场相关数据
            df = df[
                (df['Origin'].isin(self.domestic_airports)) | 
                (df['Destination'].isin(self.domestic_airports))
            ]
            
            # 标记是否为国际航班
            df['International Flights'] = df.apply(
                lambda row: 0 if (row['Origin'] in self.domestic_airports and 
                                row['Destination'] in self.domestic_airports) else 1,
                axis=1)
            df['Region'] = region

            # 分组处理 - 重命名 Avg yield 为 Avg_yield 避免关键字冲突
            df = df.rename(columns={'Avg yield': 'Avg_yield'})
            
            # 定义聚合函数（不再包含 Connecting over 相关列）
            agg_funcs = {
                'Total Est. Pax': 'sum',
                'First': 'sum',
                'Business': 'sum',
                'Premium': 'sum',
                'Full Y': 'sum',
                'Disc Y': 'sum',
                'International Flights': 'first',
                'Region': 'first'
            }

            # 计算加权平均值
            avg_columns = ['Avg_yield', 'Avg First', 'Avg Business', 'Avg Premium', 'Avg Full Y', 'Avg Disc Y']
            for col in avg_columns:
                # 获取对应的数量列名
                count_col = col.replace('Avg ', '') if col != 'Avg_yield' else 'Total Est. Pax'
                
                # 计算总价值
                df[f'{col}_total'] = df[col] * df[count_col]
                agg_funcs[f'{col}_total'] = 'sum'
                agg_funcs[count_col] = 'sum'  # 确保数量列也被求和

            # 执行分组聚合
            grouped = df.groupby(['Origin', 'Destination'])
            result = grouped.agg(agg_funcs)

            # 计算新的平均值
            for col in avg_columns:
                count_col = col.replace('Avg ', '') if col != 'Avg_yield' else 'Total Est. Pax'
                result[col] = result[f'{col}_total'] / result[count_col]
                result = result.drop(f'{col}_total', axis=1)

            # 处理除零情况
            result = result.fillna(0)
            
            # 重置索引
            result = result.reset_index()
            
            # 恢复原来的列名
            result = result.rename(columns={'Avg_yield': 'Avg yield'})
            
            # 重新排列列顺序（不包含 Connecting over 相关列）
            output_columns = self.OAG_Connection_columns
            result = result[output_columns]

            # 检查并保存结果
            output_filename = f"con_{yyyymm_str}.csv"
            output_path = os.path.join(output_folder, output_filename)
            
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, result], ignore_index=True)
                combined_df.drop_duplicates(inplace=True)
                combined_df.to_csv(output_path, index=False)
            else:
                result.to_csv(output_path, index=False)  
            # print(f"成功处理并保存: {output_path}")    
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    def process_connection(self):
        """处理oag的connection数据"""
        input_folder = self.connection_input_folder
        output_folder = self.connection_output_folder
        if os.path.exists(output_folder):
            print(f"输出目录 {output_folder} 已存在，跳过process_connection")
            return
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith('.CSV') or filename.endswith('.csv'):
                file_path = os.path.join(input_folder, filename)
                print(f"Processing connection file: {file_path}")
                self.__process_connection_and_save(file_path, output_folder)

    def __process_mix_and_save(self, file_path, output_folder):
        try:
            # 解析文件名以提取年份和月份
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            if len(parts) < 3:
                print(f"警告：文件名 {filename} 格式不正确。")
                return
            region = parts[1]  # 地区
            yyyymm = parts[2].split('_')[0]  # 年月
            yyyy, mm = int(yyyymm[:4]), int(yyyymm[4:])
            yyyymm_str = f"{yyyy:04d}-{mm:02d}"  # 格式化为 YYYY-MM

            df = pd.read_csv(file_path,index_col=False,on_bad_lines='skip')

            # 筛选从 M 到 Y 
            columns_to_keep = df.columns[:11]
            df = df[columns_to_keep]
            # 删除 Origin 列为空的行
            df = df.dropna(subset=["Market Pair"])
            # 重新命名列名
            new_column_names = self.OAG_mix_columns
            # 重新命名列名
            df.columns = new_column_names
            df[['Origin', 'Destination']] = df['Market Pair'].str.split('-', expand=True)
            df = df.drop(columns=['Market Pair']) 
            df = df[
                (df['Origin'].isin(self.domestic_airports)) | 
                (df['Destination'].isin(self.domestic_airports))
            ]
            df['International Flights'] = df.apply(
                lambda row: 0 if (row['Origin'] in self.domestic_airports and 
                                row['Destination'] in self.domestic_airports) else 1,
                axis=1)
            df['Region'] = region
            
            # 检查是否存在对应的 YYYY-MM.csv 文件
            output_filename = f"mix_{yyyymm_str}.csv"
            output_path = os.path.join(output_folder, output_filename)
            
            if os.path.exists(output_path):
                # 如果文件存在，读取文件并拼接数据
                existing_df = pd.read_csv(output_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # 去重
                combined_df.drop_duplicates(inplace=True)
                # 保存更新后的文件
                combined_df.to_csv(output_path, index=False)
            else:
                # 如果文件不存在，保存当前数据为新文件
                df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    def process_mix(self):
        """处理oag的mix数据"""
        input_folder = self.mix_input_folder
        output_folder = self.mix_output_folder
        if os.path.exists(output_folder):
            print(f"输出目录 {output_folder} 已存在，跳过process_mix")
            return
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith('.CSV') or filename.endswith('.csv'):
                file_path = os.path.join(input_folder, filename)
                print(f"Processing mix file: {file_path}")
                self.__process_mix_and_save(file_path, output_folder)
    
    def __process_monthly_data(self, file_path):
        """处理一个月的数据"""
        df = pd.read_csv(file_path)

        df = df[(df['Route_Total_Seats'] > 0)].copy()
        df['YearMonth'] = df['Time series'].str[:7]
        # 首先，计算每个月每个航线的总航班数和总座位数
        route_data = df.drop(columns=['Equipment', 'Equipment_Total_Flights','Equipment_Total_Seats'])  
        route_data = route_data.drop_duplicates()  
        route_grouped = route_data.groupby(['YearMonth', 'Origin', 'Destination'])
        route_summary = route_grouped.agg({
            'Route_Total_Flights': 'sum',
            'Route_Total_Seats': 'sum',
            'Route_Total_Flight_Time':'sum',
            'Route_Avg_Flight_Time':'mean'
        }).reset_index()
        route_summary.columns = ['YearMonth', 'Origin', 'Destination', 'Route_Total_Flights', 'Route_Total_Seats', 'Route_Total_Flight_Time', 'Route_Avg_Flight_Time']

        # 然后，计算每个月每个航线每个机型的总航班数和总座位数
        equipment_grouped = df.groupby(['YearMonth', 'Origin', 'Destination', 'Equipment', 'Distance (KM)', 'International Flights'])
        equipment_summary = equipment_grouped.agg({
            'Equipment_Total_Flights': 'sum',
            'Equipment_Total_Seats': 'sum'
        }).reset_index()
        equipment_summary.columns = ['YearMonth', 'Origin', 'Destination', 'Equipment', 'Distance (KM)', 'International Flight', 
                                    'Equipment_Total_Flights', 'Equipment_Total_Seats']

        # 将航线总数据与机型数据合并（左连接）
        df = pd.merge(equipment_summary, route_summary,
                    on=['YearMonth', 'Origin', 'Destination'], 
                    how='left')
        
        # 获取唯一的YearMonth（假设所有记录都是同一个月）
        year_month = df['YearMonth'].iloc[0]  # 取第一条记录的YearMonth

        # 构造con文件路径（例如：./Con_Processed/con_2023-01.csv）
        con_file = f"./Con_Processed/con_{year_month}.csv"
        # 定义需要从 con 文件中提取的列（必须包含 Origin 和 Destination 用于合并）
        con_required_cols = [
            'Origin', 'Destination',  # 必须包含的合并键
            'Total Est. Pax', 'First', 'Business', 'Premium', 'Full Y', 'Disc Y',
            'Avg yield', 'Avg First', 'Avg Business', 'Avg Premium', 'Avg Full Y', 'Avg Disc Y',
            'Region'
        ]

        if os.path.exists(con_file):
            # 读取 con 文件，并只选择需要的列（例如 Airline 和 Aircraft_Type）
            con_data = pd.read_csv(con_file, usecols=con_required_cols)
            # 重命名 Total Est. Pax 为 Con-Total Est. Pax
            con_data = con_data.rename(columns={'Total Est. Pax': 'Con Total Est. Pax'})
            
            # 按 Origin 和 Destination 左连接
            df = pd.merge(
                df,
                con_data,
                on=['Origin', 'Destination'],
                how='left'  # 保留 df 的所有行
            )
        else:
            print(f"错误：未找到文件 {con_file}")

        # 构造mix文件路径（例如：./Mix_Processed/mix_2023-01.csv）
        mix_file = f"./Mix_Processed/mix_{year_month}.csv"
        # 定义需要从 con 文件中提取的列（必须包含 Origin 和 Destination 用于合并）
        mix_required_cols = [
            'Origin', 'Destination',  # 必须包含的合并键
            'Total Est. Pax','Local Est. Pax','Behind Est. Pax','Bridge Est. Pax','Beyond Est. Pax',
            'Avg Fare (USD)','Local Fare','Behind Fare','Bridge Fare','Beyond Fare'
        ]

        if os.path.exists(mix_file):
            # 读取 con 文件，并只选择需要的列（例如 Airline 和 Aircraft_Type）
            mix_data = pd.read_csv(mix_file, usecols=mix_required_cols)        

            # 按 Origin 和 Destination 左连接
            df = pd.merge(
                df,
                mix_data,
                on=['Origin', 'Destination'],
                how='left'  # 保留 df 的所有行
            )
        else:
            print(f"错误：未找到文件 {mix_file}")

        return df

    def process_all_oag(self):
        """按照capacity数据,连接其他所有oag数据"""
        folder_path = self.capacity_output_folder
        # 获取文件夹中所有CSV文件
        all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        # 存储所有处理后的DataFrame
        all_dfs = []
        
        for file in all_files:
            file_path = os.path.join(folder_path, file)
            print(f"Processing oag all file: {file}")
            # 处理单个文件
            monthly_df = self.__process_monthly_data(file_path)
            all_dfs.append(monthly_df)
        
        # 纵向合并所有DataFrame
        if all_dfs:
            final_df = pd.concat(all_dfs, axis=0, ignore_index=True)
            # 按YearMonth, Origin, Destination排序
            final_df = final_df.sort_values(by=['YearMonth', 'Origin', 'Destination'])
            final_df = final_df.reset_index(drop=True)
            
            return final_df
        else:
            return pd.DataFrame()

    def __interpolate_city_data_bug(self,city_data):
        """定义一个函数，用于对每个城市的每年数据进行插值"""
        # 确保所有经济指标列为数值型，使用 pd.to_numeric 转换，并处理错误
        city_data['GDP（亿元）'] = pd.to_numeric(city_data['GDP（亿元）'], errors='coerce')
        city_data['人口（万人）'] = pd.to_numeric(city_data['人口（万人）'], errors='coerce')
        city_data['第三产业占比（%）'] = pd.to_numeric(city_data['第三产业占比（%）'], errors='coerce')
        city_data['人均可支配收入（元）'] = pd.to_numeric(city_data['人均可支配收入（元）'], errors='coerce')
        city_data['社会消费品零售总额(万元)'] = pd.to_numeric(city_data['社会消费品零售总额(万元)'], errors='coerce')
        city_data['三大产业就业人员总和(万人)'] = pd.to_numeric(city_data['三大产业就业人员总和(万人)'], errors='coerce')
        city_data['民用航空客运量(万人)'] = pd.to_numeric(city_data['民用航空客运量(万人)'], errors='coerce')
        
        # 创建一个空的列表来存储插值结果
        interpolated_data = []
        
        # 遍历每个年份
        for i, year in enumerate(city_data['年份'].unique()):
            # 获取当前年份的数据
            current_year_data = city_data[city_data['年份'] == year].iloc[0]
            
            # 如果是最后一年，直接添加到结果中
            if i == len(city_data['年份'].unique()) - 1:
                for month in range(1, 13):
                    interpolated_data.append({
                        'YearMonth': f'{year}-{month:02d}',
                        'City': current_year_data['三字码'],
                        'GDP': current_year_data['GDP（亿元）'],
                        'Population': current_year_data['人口（万人）'],
                        'Third_Indsutry': current_year_data['第三产业占比（%）'],
                        'Revenue': current_year_data['人均可支配收入（元）'],
                        'Retail': current_year_data['社会消费品零售总额(万元)'],
                        'Labor': current_year_data['三大产业就业人员总和(万人)'],
                        'Air_Traffic': current_year_data['民用航空客运量(万人)']
                    })
            else:
                # 获取下一年的数据
                next_year_data = city_data[city_data['年份'] == year + 1].iloc[0]
                
                # 对每个月进行插值
                for month in range(1, 13):
                    # 计算插值比例
                    ratio = (month - 1) / 12
                    
                    # 插值计算
                    interpolated_row = {
                        'YearMonth': f'{year}-{month:02d}',
                        'City': current_year_data['三字码'],
                        'GDP': current_year_data['GDP（亿元）'] + ratio * (next_year_data['GDP（亿元）'] - current_year_data['GDP（亿元）']),
                        'Population': current_year_data['人口（万人）'] + ratio * (next_year_data['人口（万人）'] - current_year_data['人口（万人）']),
                        'Third_Indsutry': current_year_data['第三产业占比（%）'] + ratio * (next_year_data['第三产业占比（%）'] - current_year_data['第三产业占比（%）']),
                        'Revenue': current_year_data['人均可支配收入（元）'] + ratio * (next_year_data['人均可支配收入（元）'] - current_year_data['人均可支配收入（元）']),
                        'Retail': current_year_data['社会消费品零售总额(万元)'] + ratio * (next_year_data['社会消费品零售总额(万元)'] - current_year_data['社会消费品零售总额(万元)']),
                        'Labor': current_year_data['三大产业就业人员总和(万人)'] + ratio * (next_year_data['三大产业就业人员总和(万人)'] - current_year_data['三大产业就业人员总和(万人)']),
                        'Air_Traffic': current_year_data['民用航空客运量(万人)'] + ratio * (next_year_data['民用航空客运量(万人)'] - current_year_data['民用航空客运量(万人)'])
                    }
                    interpolated_data.append(interpolated_row)
        
        # 将插值数据转换为DataFrame并返回
        return pd.DataFrame(interpolated_data)

    def __interpolate_city_data(self, city_data):
        """对每个城市的每年数据进行插值和预测，处理中间年份缺失问题"""
        # 确保所有经济指标列为数值型
        indicators = ['GDP（亿元）', '人口（万人）', '第三产业占比（%）', 
                    '人均可支配收入（元）', '社会消费品零售总额(万元)', 
                    '三大产业就业人员总和(万人)', '民用航空客运量(万人)']
        
        for indicator in indicators:
            city_data[indicator] = pd.to_numeric(city_data[indicator], errors='coerce')
        
        # ==== 新增：处理中间年份缺失问题 ====
        # 获取年份范围
        years = sorted(city_data['年份'].unique())
        min_year = min(years)
        max_year = max(years)
        
        # 创建完整年份范围的数据框架
        all_years = pd.DataFrame({
            '年份': range(min_year, max_year + 1),
            '三字码': city_data['三字码'].iloc[0]  # 保持城市代码一致
        })
        
        # 合并原始数据，填充缺失年份
        city_data = pd.merge(all_years, city_data, how='left', on=['年份', '三字码'])
        
        # 处理缺失值和0值
        for indicator in indicators:
            # 将0值视为缺失
            city_data.loc[city_data[indicator] == 0, indicator] = np.nan
            
            # 使用线性插值填充中间年份
            city_data[indicator] = city_data[indicator].interpolate(method='linear')
            
            # 如果两端有缺失，用最近值填充
            city_data[indicator] = city_data[indicator].fillna(method='ffill').fillna(method='bfill')

            if city_data[indicator].isna().any():
                city_data[indicator] = city_data[indicator].fillna(0)
        # ==== 修复完成 ====
        
        # 创建一个空的列表来存储插值结果
        interpolated_data = []
        
        # 重新获取年份范围（已包含完整年份）
        years = sorted(city_data['年份'].unique())
        min_year = min(years)
        max_year = max(years)
        
        # 预测下一年数据（用于月度插值）
        next_year_data = {}
        for indicator in indicators:
            # 使用所有年份数据训练模型
            X = city_data['年份'].values.reshape(-1, 1)
            y = city_data[indicator].values

            # 如果y包含NaN，使用简单平均值替代
            if np.isnan(y).any():
                mean_val = np.nanmean(y)
                y = np.where(np.isnan(y), mean_val, y)

                # 如果所有值都是NaN，使用0
                if np.isnan(mean_val):
                    y = np.zeros_like(y)
            
            model = LinearRegression()
            model.fit(X, y)
            next_year_value = model.predict([[max_year + 1]])[0]
            next_year_data[indicator] = next_year_value
        
        # 对每个年份进行月度插值
        for year in years:
            current_data = city_data[city_data['年份'] == year].iloc[0]
            
            # 确定下一年数据
            if year < max_year:
                next_data = city_data[city_data['年份'] == year + 1].iloc[0]
            else:
                # 使用预测的下一年数据
                next_data = {'年份': year + 1, '三字码': current_data['三字码']}
                for indicator in indicators:
                    next_data[indicator] = next_year_data[indicator]
            
            # 按月插值
            for month in range(1, 13):
                row = {'YearMonth': f'{year}-{month:02d}', 'City': current_data['三字码']}
                
                for indicator in indicators:
                    # 线性插值公式
                    ratio = (month - 1) / 12.0
                    current_val = current_data[indicator]
                    next_val = next_data[indicator]
                    interpolated_val = current_val + ratio * (next_val - current_val)
                    
                    row[indicator] = interpolated_val
                
                interpolated_data.append(row)
        
        # 转换为DataFrame并重命名列
        result = pd.DataFrame(interpolated_data)
        col_mapping = {
            'GDP（亿元）': 'GDP',
            '人口（万人）': 'Population',
            '第三产业占比（%）': 'Third_Industry',
            '人均可支配收入（元）': 'Revenue',
            '社会消费品零售总额(万元)': 'Retail',
            '三大产业就业人员总和(万人)': 'Labor',
            '民用航空客运量(万人)': 'Air_Traffic'
        }
        return result.rename(columns=col_mapping)

    def process_economy(self):
        ecodata = pd.read_csv(self.economy_file)
        # 对每个城市分别进行插值
        eco_df = pd.DataFrame()
        for city in ecodata['三字码'].unique():
            city_data = ecodata[ecodata['三字码'] == city]
            interpolated_data = self.__interpolate_city_data(city_data)
            eco_df = pd.concat([eco_df, interpolated_data], ignore_index=True)
        # 直接修改 eco_df 的列名
        eco_df_origin = eco_df.rename(columns={
            'City': 'Origin',
            'GDP': 'O_GDP',
            'Population': 'O_Population',
            'Third_Indsutry': 'O_Third_Indsutry',
            'Revenue': 'O_Revenue',
            'Retail': 'O_Retail',
            'Labor': 'O_Labor',
            'Air_Traffic': 'O_Air_Traffic'
        })
        # Step 2: 按照 YearMonth 和 Destination 拼接
        # 直接修改 eco_df 的列名
        eco_df_destination = eco_df.rename(columns={
            'City': 'Destination',
            'GDP': 'D_GDP',
            'Population': 'D_Population',
            'Third_Indsutry': 'D_Third_Indsutry',
            'Revenue': 'D_Revenue',
            'Retail': 'D_Retail',
            'Labor': 'D_Labor',
            'Air_Traffic': 'D_Air_Traffic'
        })

        return eco_df_origin,eco_df_destination

    def process_data(self):
        """主处理流程"""
        # 处理所有OAG数据
        self.process_capacity()
        self.process_connection()
        self.process_mix()
        OAG_df = self.process_all_oag()
        # 处理经济数据
        eco_df_origin,eco_df_destination = self.process_economy()
        # 对 Origin 进行合并，合并后的列名已加上 O_ 前缀
        OAG_df['YearMonth'] = OAG_df['YearMonth'].apply(lambda x: str(x))  # 确保格式一致
        eco_df_origin['YearMonth'] = eco_df_origin['YearMonth'].apply(lambda x: str(x))  # 确保格式一致
        final_df = pd.merge(OAG_df, eco_df_origin, how='left', left_on=['YearMonth', 'Origin'], right_on=['YearMonth', 'Origin'])

        eco_df_destination['YearMonth'] = eco_df_destination['YearMonth'].apply(lambda x: str(x))  # 确保格式一致
        final_df = pd.merge(final_df, eco_df_destination, how='left', left_on=['YearMonth', 'Destination'], right_on=['YearMonth', 'Destination'])

        final_df.to_csv(self.final_output_file,index=False)


if __name__ == "__main__":
    airport_file = "./中国机场代码.xlsx"
    economy_file = './城市经济数据.csv'
    capacity_input_folder = './OAG_Capacity/OAG_Capacity_Report(1996-2024.6.30)/capacity1997-2024'
    connection_input_folder = './OAG_Traffic/OAG_Traffic_Report(2010-2024)/traffic2011-2024/con2011-2024'
    mix_input_folder = './OAG_Traffic/OAG_Traffic_Report(2010-2024)/traffic2011-2024/mix2011-2024'
    final_output_file = './final_data_0623.csv'
    processor = DataCleaner(airport_file=airport_file, 
                            economy_file=economy_file, 
                            capacity_input_folder=capacity_input_folder, 
                            connection_input_folder=connection_input_folder, 
                            mix_input_folder=mix_input_folder, 
                            final_output_file=final_output_file)
    processor.process_data()



