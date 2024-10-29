import pandas as pd
import matplotlib.pyplot as plt
import pytz
from pytz import country_timezones
import seaborn as sns

# 读取数据
data = pd.read_csv('user_activity_with_location.csv', parse_dates=['activity_date', 'activity_time'])

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


# 获取每个国家的时区，默认使用第一个时区
def get_timezone(country):
    if pd.isna(country):  # 检查国家字段是否为空
        return 'Asia/Shanghai'  # 默认使用中国时区
    try:
        timezone = country_timezones[country][0]  # 选择该国家的第一个时区
    except KeyError:
        timezone = 'Asia/Shanghai'  # 如果找不到该国家的时区，默认使用中国时间
    return timezone

# 自动获取时区并转换为本地时间
def convert_to_local_time(row):
    country_code = row['country']
    timezone_str = get_timezone(country_code)  # 根据国家代码获取时区
    china_time = row['activity_time']  # 原始数据已经是中国时间
    if china_time.tzinfo is None:  # 检查是否有时区信息
        china_time = china_time.tz_localize('Asia/Shanghai')  # 标记为中国时区
    local_time = china_time.astimezone(pytz.timezone(timezone_str))  # 转换为当地时区
    return local_time

# 转换活动时间为当地时间，并提取日期、小时和星期几
data['local_time'] = data.apply(convert_to_local_time, axis=1)
data['local_hour'] = data['local_time'].dt.hour

# 清洗国家名，确保没有前后空格且统一大小写
data['country'] = data['country'].str.strip().str.title()

# 选择活动最多的前10个国家
top_countries = data['country'].value_counts().nlargest(10).index
top_country_data = data[data['country'].isin(top_countries)]

# 按小时和国家汇总活动数据
country_hour_activity = top_country_data.groupby(['local_hour', 'country'])['activity_count'].sum().unstack(fill_value=0)

# 绘制不同国家在一天中的活跃时间段（类似上传的图）
plt.figure(figsize=(12, 8))
colors = sns.color_palette("tab10", len(top_countries))  # 使用Seaborn的颜色调色板
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']  # 定义不同的线条样式

# 绘制每个国家的线条
for i, country in enumerate(country_hour_activity.columns):
    plt.plot(country_hour_activity.index, country_hour_activity[country], 
             label=country, color=colors[i], linestyle=line_styles[i % len(line_styles)], linewidth=2)

# 优化图形细节
plt.title('不同国家用户活跃时段对比（当地时间）', fontsize=16, fontweight='bold')
plt.xlabel('小时', fontsize=14)
plt.ylabel('活动总数', fontsize=14)
plt.xticks(range(0, 24), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='国家', fontsize=12, title_fontsize=14, loc='upper right')
plt.tight_layout()
plt.show()
