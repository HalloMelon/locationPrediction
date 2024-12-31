import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the dataset
file_path = 'ML_predict/combined_users_with_predictions_cleaned.csv'
data = pd.read_csv(file_path)

# Identify all features except the target column 'country'
# features = data.drop(columns=['country'])
#  data[[fullname,email,all_activity_count,utc_offset,most_active_hour,user_location,country,first_name,last_name,predicted_country,probability]]
features = data.drop(columns=['country','fullname', 'user_location'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['country'])

X_combined = features.astype(str).agg(' '.join, axis=1)

# Tokenizing the combined input data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_combined)
X_sequences = tokenizer.texts_to_sequences(X_combined)
X_padded = pad_sequences(X_sequences, padding='post')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.3, random_state=42)

# Define the BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X_padded.shape[1]))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.3)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# Predict on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate evaluation metrics
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Display the results
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')