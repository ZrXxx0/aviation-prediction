import pandas as pd
import numpy as np

def filter_routes(min_ratio, report_path, span_ratio_threshold=0.9):
    """
    筛选满足条件的航线：
    1. 有效数据占比(Valid_Ratio)不低于给定阈值
    2. 时间跨度达到阈值
    
    参数:
    min_ratio - 有效数据占比阈值 (0.0到1.0之间)
    report_path - 航线报告文件路径
    
    返回:
    valid_routes - 满足条件的航线DataFrame
    """
    # 1. 读取航线报告
    route_report = pd.read_csv(report_path)
    
    # 2. 预处理年月列 - 提取YYYY-MM部分
    # 确保列存在且不为空
    if 'Min_YearMonth' in route_report.columns and 'Max_YearMonth' in route_report.columns:
        # 转换为字符串并提取前7个字符（YYYY-MM）
        route_report['Min_YearMonth'] = route_report['Min_YearMonth'].astype(str).str[:7]
        route_report['Max_YearMonth'] = route_report['Max_YearMonth'].astype(str).str[:7]
    else:
        print("错误: 报告中缺少Min_YearMonth或Max_YearMonth列")
        return pd.DataFrame()
    
    # 3. 转换时间列为日期类型并计算时间跨度
    route_report['Min_Date'] = pd.to_datetime(
        route_report['Min_YearMonth'], format='%Y-%m', errors='coerce'
    )
    route_report['Max_Date'] = pd.to_datetime(
        route_report['Max_YearMonth'], format='%Y-%m', errors='coerce'
    )
    
    # 4. 计算时间跨度(月)
    # 处理可能的NaN值
    days_diff = (route_report['Max_Date'] - route_report['Min_Date']).dt.days
    days_diff = days_diff.fillna(0)  # 填充NaN为0
    
    # 计算月份跨度（确保非负）
    route_report['Month_Span'] = (days_diff / 30).round().astype(int)
    route_report['Month_Span'] = route_report['Month_Span'].clip(lower=0)  # 确保非负
    
    # 5. 过滤有效比例大于阈值的航线
    valid_routes = route_report[route_report['Valid_Ratio'] >= min_ratio].copy()
    
    # 6. 如果没有满足条件的航线，直接返回空DataFrame
    if valid_routes.empty:
        print(f"警告: 没有找到有效比例 ≥ {min_ratio} 的航线")
        return valid_routes
    
    # 7. 找出最大时间跨度（忽略0值）
    max_span = valid_routes[valid_routes['Month_Span'] > 0]['Month_Span'].max()
    
    # 处理没有有效时间跨度的情况
    if pd.isna(max_span):
        print(f"警告: 没有找到有效的时间跨度数据")
        return pd.DataFrame()  # 返回空DataFrame
    
    print(f"最大时间跨度: {max_span} 个月")
    print(f"时间跨度阈值: {max_span * span_ratio_threshold:.1f} 个月 (最大跨度的 {span_ratio_threshold*100}%)")
    
    # 8. 计算最小要求的时间跨度
    min_span_threshold = max_span * span_ratio_threshold
    
    # 9. 筛选时间跨度达到阈值的航线
    final_routes = valid_routes[valid_routes['Month_Span'] >= min_span_threshold].copy()
    
    # 10. 按质量排序
    final_routes = final_routes.sort_values(
        by=['Valid_Ratio', 'Month_Span'], 
        ascending=[False, False]
    )
    
    # 11. 添加时间跨度信息
    final_routes['Time_Span'] = final_routes['Min_YearMonth'] + " to " + final_routes['Max_YearMonth']
    
    # 12. 重置索引
    final_routes.reset_index(drop=True, inplace=True)
    
    # # 13. 打印统计信息
    # print(f"筛选出 {len(final_routes)} 条高质量航线")
    
    # if not final_routes.empty:
    #     # 获取时间范围
    #     min_date = final_routes['Min_Date'].min()
    #     max_date = final_routes['Max_Date'].max()
        
    #     print(f"时间范围: {min_date.strftime('%Y-%m')} 至 {max_date.strftime('%Y-%m')}")
    #     print(f"平均有效数据比例: {final_routes['Valid_Ratio'].mean():.2%}")
    #     print(f"平均时间跨度: {final_routes['Month_Span'].mean():.1f} 个月")
    #     print(f"时间跨度范围: {final_routes['Month_Span'].min()} - {final_routes['Month_Span'].max()} 个月")
    
    return final_routes