import pandas as pd
from collections import defaultdict
import re

# Load the names.tab file into a DataFrame
names_tab_path = 'names.tab'  # 请将路径修改为您的文件实际路径
names_df = pd.read_csv(names_tab_path, sep='\t', header=None, names=['name', 'type', 'country', 'frequency'])

# Prepare dictionaries for storing probabilities
forenames_dict = defaultdict(lambda: defaultdict(float))
surnames_dict = defaultdict(lambda: defaultdict(float))

# Fill dictionaries with probabilities based on frequency
for index, row in names_df.iterrows():
    name = str(row['name']).lower()
    country = row['country']
    frequency = row['frequency']
    if row['type'] == 'forename':
        forenames_dict[name][country] += frequency
    else:
        surnames_dict[name][country] += frequency

# Normalize probabilities so they sum to 1
for name, countries in forenames_dict.items():
    total_freq = sum(countries.values())
    for country in countries:
        forenames_dict[name][country] /= total_freq

for name, countries in surnames_dict.items():
    total_freq = sum(countries.values())
    for country in countries:
        surnames_dict[name][country] /= total_freq

# Load the CSV file with names to be predicted
csv_file_path = 'ML_predict/combined_users.csv'
users_df = pd.read_csv(csv_file_path)

# Ensure 'fullname' column is treated as string and handle missing values
users_df['fullname'] = users_df['fullname'].fillna('').astype(str)
split_names = users_df['fullname'].apply(lambda x: re.split(r'\s+', x.strip(), maxsplit=1))
users_df['first_name'] = split_names.apply(lambda x: x[0].lower())
users_df['last_name'] = split_names.apply(lambda x: x[1].lower() if len(x) > 1 else '')

# Predict country for each name
def predict_country(first_name, last_name):
    country_probs = defaultdict(float)
    
    # Get probabilities from forenames_dict
    if first_name in forenames_dict:
        for country, prob in forenames_dict[first_name].items():
            country_probs[country] += prob
    
    # Get probabilities from surnames_dict
    if last_name in surnames_dict:
        for country, prob in surnames_dict[last_name].items():
            country_probs[country] += prob
    
    # Normalize probabilities so they sum to 1
    total_prob = sum(country_probs.values())
    if total_prob > 0:
        for country in country_probs:
            country_probs[country] /= total_prob
    
    # Get the most probable country and its probability
    if country_probs:
        predicted_country = max(country_probs, key=country_probs.get)
        probability = country_probs[predicted_country]
        return predicted_country, probability
    else:
        return 'Unknown', 0.0

# Apply prediction to each row
users_df[['predicted_country', 'probability']] = users_df.apply(
    lambda row: pd.Series(predict_country(row['first_name'], row['last_name'])), axis=1
)

# Save the updated DataFrame to a new CSV file
output_file_path = 'ML_predict/combined_users_with_predictions.csv'
users_df.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")