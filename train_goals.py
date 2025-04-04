import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

def main():
    # 1. Load the merged file with updated path
    data_path = "data/2024-25/gws/merged_gw_cleaned.csv"  # Updated path to cleaned CSV
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        print("Please run the CSV cleaner script first to create the cleaned CSV file.")
        return

    try:
        data = pd.read_csv(data_path, low_memory=False)
        print(f"Successfully loaded data with shape: {data.shape}")
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return

    print("Columns in merged_gw_cleaned.csv:", data.columns.tolist())
    print("Sample rows:\n", data.head())

    # 2. Update training data to use GW 1-30
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 30)]

    # Check if training data is available
    if train_data.empty:
        print("No training data found for GW 1-30 in merged_gw_cleaned.csv.")
        return

    print(f"Training data shape (GW 1-30): {train_data.shape}")

    # 3. Select features & target
    features = [
        "minutes",       # How long the player played
        "xP",            # Expected points (given in your CSV)
        "threat",        # Attacking threat
        "bps",           # Bonus points system
        "transfers_in",  # Possibly a proxy for form
        "expected_goals" # Another good feature if your merged_gw.csv includes it
    ]

    # Check if all features exist
    missing_features = [f for f in features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        # Remove missing features from the list
        features = [f for f in features if f in train_data.columns]
        print(f"Using these features instead: {features}")

    target = "goals_scored"

    # Drop rows with missing values in the chosen features/target
    train_data = train_data.dropna(subset=features + [target])

    # Convert columns to numeric if needed
    for col in features + [target]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 4. Split training data into train/test sets for evaluation
    X = train_data[features]
    y = train_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate the model on the test set
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print("Mean Absolute Error (MAE) on the test set:", mae)

    # 7. Prepare GW 31 data for prediction
    # Group by player and team, but only include numeric columns for averaging
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    gw31_data = train_data.groupby(["name", "team"])[numeric_columns].mean().reset_index()
    gw31_data["GW"] = 31  # Updated to GW 31

    # Ensure the features are present in the GW 31 data
    X_future = gw31_data[features]

    # 8. Predict goals for GW 31
    gw31_data["predicted_goals"] = model.predict(X_future)

    # 9. Filter out rows with predicted_goals equal to 0
    gw31_data = gw31_data[gw31_data["predicted_goals"] > 0]

    # 10. Sort by predicted_goals in descending order
    gw31_data_sorted = gw31_data.sort_values(by="predicted_goals", ascending=False)

    # 11. Save the sorted predictions to a new file
    out_path = "GW_highest_Predicted_goals.csv"
    gw31_data_sorted.to_csv(out_path, index=False)
    print(f"Sorted predictions (without 0 values) for GW 31 saved to {out_path}")
    print("Sample predictions:\n", gw31_data_sorted[["name", "team", "GW", "predicted_goals"]].head(10))

if __name__ == "__main__":
    main()