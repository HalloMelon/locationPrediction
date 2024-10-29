# 一、处理sql跑出来的原始数据，处理utc
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
import re
from csv import writer

# 第一步：加载数据集并提取时间戳
import pandas as pd

# 读取CSV数据
data = pd.read_csv('top_users/2_activity_info_from_sql/United States.csv')

# 提取时间部分，将字符串转换为datetime对象
data['activity_time'] = pd.to_datetime(data['activity_time'], utc=True)

# 将时间转换为小时
data['hour'] = data['activity_time'].dt.hour

# 生成每个用户的UTC偏移量数组
def generate_utc_offset_distribution(group):
    hours = group['hour'].values
    distribution = [0] * 24
    for hour in hours:
        distribution[hour] += 1
    return distribution

# 为每个用户生成UTC偏移量分布矩阵
utc_offset_matrix = data.groupby('commit_id').apply(generate_utc_offset_distribution)

# 将分布矩阵转换为DataFrame
utc_offset_df = pd.DataFrame(utc_offset_matrix.tolist(), index=utc_offset_matrix.index)

# # 将分布矩阵导出到文件
# utc_offset_df.to_csv('utc_offset_distribution.csv', index_label='commit_id')

# print("UTC偏移量分布矩阵已导出到文件 utc_offset_distribution.csv")

# 按用户ID分组并统计活跃时间
grouped = data.groupby('commit_id')['hour'].agg(lambda x: x.value_counts().idxmax())
data = data.merge(grouped, on='commit_id', suffixes=('', '_local_peak_hour'))

# 计算UTC偏移量（假设最频繁的活动时间是开发者的本地白天时间）
def infer_utc_offset(local_peak_hour):
    # 假设9点到17点是本地工作时间，如果高频时间是1点，我们认为可能是偏移 -8 小时（即美国西海岸）
    if 9 <= local_peak_hour <= 17:
        return 0  # 已经是白天，不需要偏移
    elif local_peak_hour < 9:
        return -(9 - local_peak_hour)  # 需要向后偏移，确保在白天
    else:
        return 24 - local_peak_hour + 9  # 向前调整，确保在白天

data['utc_offset'] = data['hour_local_peak_hour'].apply(infer_utc_offset)

# # 将结果保存到文件
# data[['commit_id', 'activity_time', 'utc_offset']].to_csv('user_utc_offset.csv', index=False)
# print("用户UTC偏移量数据已导出到文件 user_utc_offset.csv")

# 增加特征值：提交时间的标准偏差和最频繁活动的时区偏移量
# 提交时间的标准偏差
std_offset = data.groupby('commit_id')['hour'].std().reset_index()
std_offset.columns = ['commit_id', 'hour_std']
data = data.merge(std_offset, on='commit_id', how='left')

# 提交时间的平均值
mean_offset = data.groupby('commit_id')['hour'].mean().reset_index()
mean_offset.columns = ['commit_id', 'hour_mean']
data = data.merge(mean_offset, on='commit_id', how='left')

# 增加特征值：每个用户的活跃天数分布
# 提取活跃的星期几
data['day_of_week'] = data['activity_time'].dt.dayofweek

# 计算每个用户在一周中的活跃分布
active_days = data.groupby('commit_id')['day_of_week'].value_counts().unstack(fill_value=0).reset_index()
active_days.columns = ['commit_id'] + [f'day_{i}' for i in range(7)]
data = data.merge(active_days, on='commit_id', how='left')

# 将所有处理后的数据写入CSV文件
data.to_csv('top_users/3_activity_processed/United States_user_activity_data.csv', index=False)
print("utc转换完成")




# # 二：进一步处理数据

# import pandas as pd
# # 读取CSV文件
# data = pd.read_csv('to_visualize/processed_user_activity_data_1.csv')

# # 删除多列，使用列表来指定要删除的列
# data = data.drop(['activity_date', 'user_location', 'longitude', 'latitude', 'hour', 'hour_std', 'hour_mean', 'day_of_week', 'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6'], axis=1)

# # # 去重，基于 'commit_id' 列
# # data = data.drop_duplicates(subset=['commit_id'])

# # 修改列名 'activity_time' 为 'datetime'
# data = data.rename(columns={'activity_time': 'datetime'})

# # 去掉 'datetime' 列中最后的 '+00:00'
# data['datetime'] = data['datetime'].str.replace(r'\+00:00', '', regex=True)

# # 保存到新的CSV文件
# data.to_csv('pre_process_data_results_2.csv', index=False)



# 另：处理llm生成的数据
# import pandas as pd

# # Load the data
# data = pd.read_csv('llm_extracted_data.csv')

# # Remove rows without a clear country prediction
# data = data[~data['country'].str.contains("Based on the information provided", na=False)]
# data = data[~data['country'].str.contains("I'm sorry", na=False)]
# data = data[~data['country'].str.contains("To predict", na=False)]
# data = data[~data['country'].str.contains("As an AI", na=False)]
# data = data[~data['country'].str.contains("is", na=False)]
# # Standardize country names
# data['country'] = data['country'].replace({
#     'The United States': 'United States',
#     'USA': 'United States',
#     'The Netherlands': 'Netherlands'
# })

# # Count deleted rows
# deleted_rows_count = len(pd.read_csv('llm_extracted_data.csv')) - len(data)

# # Save the cleaned data to a new CSV file
# data.to_csv('cleaned_llm_extracted_data.csv', index=False)

# # Output the count of deleted rows
# deleted_rows_count

# import pandas as pd
# data=pd.read_csv('cleaned_llm_extracted_data.csv', encoding='ISO-8859-1')
# # 去重，保留第一次出现的行
# data = data.drop_duplicates()
# # 保存到新的CSV文件
# data.to_csv('cleaned_llm_extracted_data1.csv', index=False)
