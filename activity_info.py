import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Load your CSV file
data = pd.read_csv('combined_users_activity_info.csv')

# Convert activity_time to datetime format and extract hour, day of the week, and date
data['activity_time'] = pd.to_datetime(data['activity_time'])
data['hour'] = data['activity_time'].dt.hour
data['day_of_week'] = data['activity_time'].dt.dayofweek
data['date'] = data['activity_time'].dt.date

# Create a pivot table for the heatmap
daily_activity = data.pivot_table(index='hour', columns='day_of_week', values='activity_count', aggfunc=np.sum, fill_value=0)

# Create a mapping for day of the week to Chinese labels
days_labels = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期天']
daily_activity.columns = [days_labels[day] for day in daily_activity.columns]

# Set font properties to ensure Chinese characters are displayed correctly
prop = fm.FontProperties(family='SimSun')  # Use SimSun to ensure Chinese characters are displayed in Song font

# Plot the heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(daily_activity, cmap='YlGnBu', annot=False, cbar=True)
plt.title('一周内用户活跃时间热力图（当地时间）', fontsize=16, fontproperties=prop)
plt.xlabel('星期', fontsize=14, fontproperties=prop)
plt.ylabel('小时', fontsize=14, fontproperties=prop)
plt.xticks(fontsize=12, fontproperties=prop)
plt.yticks(fontsize=12, fontproperties=prop)
plt.show()

# Additional visualization: Daily total user activity line plot
daily_total_activity = data.groupby('date')['activity_count'].sum().reset_index()
plt.figure(figsize=(16, 6))
plt.plot(daily_total_activity['date'], daily_total_activity['activity_count'], marker='o')
plt.title('每日用户活动总量', fontsize=16, fontproperties=prop)
plt.xlabel('日期', fontsize=14, fontproperties=prop)
plt.ylabel('活动总量', fontsize=14, fontproperties=prop)
plt.xticks(rotation=45, fontsize=12, fontproperties=prop)
plt.yticks(fontsize=12, fontproperties=prop)
plt.grid(True)
plt.show()

# Additional visualization: Hourly activity distribution
hourly_activity = data.groupby('hour')['activity_count'].sum().reset_index()
plt.figure(figsize=(16, 6))
plt.bar(hourly_activity['hour'], hourly_activity['activity_count'], color='skyblue')
plt.title('每小时用户活动分布', fontsize=16, fontproperties=prop)
plt.xlabel('小时', fontsize=14, fontproperties=prop)
plt.ylabel('活动总量', fontsize=14, fontproperties=prop)
plt.xticks(fontsize=12, fontproperties=prop)
plt.yticks(fontsize=12, fontproperties=prop)
plt.grid(axis='y')
plt.show()

# Additional visualization: Weekly user activity distribution
weekly_activity = data.groupby('day_of_week')['activity_count'].sum().reset_index()
weekly_activity['day_of_week'] = weekly_activity['day_of_week'].apply(lambda x: days_labels[x])
plt.figure(figsize=(16, 6))
plt.bar(weekly_activity['day_of_week'], weekly_activity['activity_count'], color='lightgreen')
plt.title('每周用户活动分布', fontsize=16, fontproperties=prop)
plt.xlabel('星期', fontsize=14, fontproperties=prop)
plt.ylabel('活动总量', fontsize=14, fontproperties=prop)
plt.xticks(fontsize=12, fontproperties=prop)
plt.yticks(fontsize=12, fontproperties=prop)
plt.grid(axis='y')
plt.show()

# Additional visualization: Activity count box plot by day of the week
plt.figure(figsize=(16, 8))
sns.boxplot(x='day_of_week', y='activity_count', data=data, palette='Set3')
plt.title('每周各天用户活动量分布箱线图', fontsize=16, fontproperties=prop)
plt.xlabel('星期', fontsize=14, fontproperties=prop)
plt.ylabel('活动量', fontsize=14, fontproperties=prop)
plt.xticks(ticks=range(7), labels=days_labels, fontsize=12, fontproperties=prop)
plt.yticks(fontsize=12, fontproperties=prop)
plt.grid(axis='y')
plt.show()
