import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

# Read the CSV (setting low_memory=False to suppress warnings)
data = pd.read_csv('data/cleaned_merged_seasons.csv', low_memory=False)

print("Columns in the dataset:", data.columns.tolist())

# Define your desired features and target (update this list to match the model)
features = ['total_points', 'selected', 'goals_scored', 'assists', 'bps', 'clean_sheets', 'influence', 'transfers_in', 'value']
target = 'minutes'

# Drop rows with missing values in selected features + target
data = data.dropna(subset=features + [target])

# Convert numeric columns to proper numbers
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Create X (features) and y (target)
X = data[features]
y = data[target]

# Print to confirm feature alignment
print(f"Dataset prepared with {X.shape[1]} features. Expected: {len(features)}")
print("Feature columns used:", X.columns.tolist())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate model performance
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

# Save the trained model and feature list
with open('minutes_rf_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': features}, f)

print("Model training complete. Model and feature names saved as minutes_rf_model.pkl")
