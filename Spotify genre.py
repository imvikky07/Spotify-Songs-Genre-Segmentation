# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:12:53 2024

@author: Vivek
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv('SpotifyFeatures.csv')

# Step 2: Data Preprocessing
# Handle missing values
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
df['genre'] = le.fit_transform(df['genre'])

# Select features and target variable
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
X = df[features]
y = df['genre']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Train various machine learning models
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42)
}

# Dictionary to store the results
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

# Step 5: Print the results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")

# Step 6: Example prediction
example_song = np.array([[0.5, 0.7, 0.6, 0.0, 0.1, -5.0, 0.05, 120.0, 0.5]])
example_song_scaled = scaler.transform(example_song)
best_model = models['Random Forest']  # Assuming Random Forest performed best
prediction = best_model.predict(example_song_scaled)
predicted_genre = le.inverse_transform(prediction)
print(f"Predicted Genre for example song: {predicted_genre[0]}")
