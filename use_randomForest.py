import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Set font to handle both English and Chinese characters
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'SimSun']  # Use Times New Roman with fallback to SimHei or SimSun for Chinese
matplotlib.rcParams['axes.unicode_minus'] = False  # To properly display the minus sign

# Load the dataset
file_path = 'combined_users_with_predictions_cleaned.csv'
data = pd.read_csv(file_path)

# Identify all features except the target column 'country'
X = data[['fullname', 'email', 'all_activity_count', 'utc_offset', 'most_active_hour', 'user_location','predicted_country', 'probability']]
y = data['country']

# Convert categorical features to numerical values if needed
X = pd.get_dummies(X)

# Encode the target variable
y = LabelEncoder().fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display the results
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Feature importance analysis
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importances})

# Aggregate the importance of all 'predicted_country' and 'user_location' related features
importance_df['特征'] = importance_df['特征'].str.replace(r'predicted_country.*', 'predicted_country', regex=True)
importance_df['特征'] = importance_df['特征'].str.replace(r'user_location.*', 'user_location', regex=True)
predicted_country_count = (importance_df['特征'] == 'predicted_country').sum()
user_location_count = (importance_df['特征'] == 'user_location').sum()
importance_df = importance_df.groupby('特征', as_index=False).agg({'重要性': 'sum'})
importance_df.loc[importance_df['特征'] == 'predicted_country', '重要性'] /= predicted_country_count
importance_df.loc[importance_df['特征'] == 'user_location', '重要性'] /= user_location_count

# Scale importance values to make them more visually distinguishable
importance_df['重要性'] *= 10

# Sort by importance
importance_df = importance_df.sort_values(by='重要性', ascending=False)

# Select top 5 features for better visualization
top_features_df = importance_df.head(5)

# Plot top 5 feature importances using a bar chart with different black and white patterns
plt.figure(figsize=(10, 6))

# Define patterns
patterns = ['+', '\\', '|', '-', '/']

bars = plt.barh(top_features_df['特征'], top_features_df['重要性'], height=0.5, color='white', edgecolor='black')  # height 设置为 0.4 使条形更细

for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

# 标题与标签
plt.title('前五个最重要的特征', fontsize=18, fontweight='bold', family='SimSun')  # 使用宋体，模拟加粗效果
plt.xlabel('重要性', fontsize=16, fontweight=900, family='SimSun')
plt.xticks(fontsize=14, fontweight='bold', family='SimSun')

# 调整图例字体大小
legend_patches = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=pattern, label=feature) 
                  for pattern, feature in zip(patterns, top_features_df['特征'])]
plt.legend(handles=legend_patches, loc='upper right', prop={'size': 18, 'family': 'Times New Roman'})  # 使用宋体，字体大小为18

plt.tight_layout()
plt.show()