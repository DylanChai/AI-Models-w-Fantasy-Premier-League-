import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

# Load the merged gameweek data
url_merged_gw = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"
response = requests.get(url_merged_gw)
df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')

# Load the players data
url_players = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/players_raw.csv"
response = requests.get(url_players)
players_df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')

# Print column names for debugging
print("Columns in df:", df.columns)
print("Columns in players_df:", players_df.columns)

# Merge the dataframes
df = pd.merge(df, players_df[['id', 'element_type', 'team', 'now_cost']], left_on='element', right_on='id', how='left')

# Print column names after merge for debugging
print("Columns after merge:", df.columns)

# Check if 'team' column exists and print some values
if 'team' in df.columns:
    print("Sample 'team' values:", df['team'].head())
else:
    print("'team' column not found after merge")

# Calculate form (average goals in last 5 gameweeks)
df['form'] = df.groupby('name')['goals_scored'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

# Select relevant features
features = [
    'minutes',
    'expected_goals',
    'shots',
    'shots_on_target',
    'form',
    'element_type',
    'ict_index',
    'threat',
    'influence',
    'creativity',
    'expected_assists',
    'expected_goal_involvements',
]

# Encode categorical variables if 'team' column exists
if 'team' in df.columns:
    le = LabelEncoder()
    df['team_encoded'] = le.fit_transform(df['team'])
    features.append('team_encoded')
else:
    print("Warning: 'team' column not found, proceeding without team information")

# Prepare the data
X = df[features]
y = df['goals_scored']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prepare data for GW25 prediction
gw25_data = df[df['GW'] == 24].copy()  # Use GW24 data as a base for GW25
gw25_data['GW'] = 25

# Make predictions for GW25
gw25_predictions = model.predict(gw25_data[features])

# Add predictions to the dataframe
gw25_data['predicted_goals'] = gw25_predictions

# Select relevant columns for output
output_columns = ['name', 'element_type', 'now_cost', 'form', 'predicted_goals']
if 'team' in df.columns:
    output_columns.insert(1, 'team')

# Create the output dataframe
output_df = gw25_data[output_columns].copy()

# Round predicted goals to 2 decimal places
output_df['predicted_goals'] = output_df['predicted_goals'].round(2)

# Sort by predicted goals in descending order
output_df = output_df.sort_values('predicted_goals', ascending=False)

# Save to CSV
output_df.to_csv('Updated_Predicted_Goals.csv', index=False)

print("Predictions saved to Updated_Predicted_Goals.csv")

# Print feature importances
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print("\nFeature importances:")
print(feature_importance.sort_values('importance', ascending=False))
