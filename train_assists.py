import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # 1. Load the merged gameweek file for the 2024-25 season.
    data_path = "data/2024-25/gws/merged_gw_cleaned.csv"  # Updated path to use cleaned CSV
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

    # 2. Use gameweeks 1-24 as training data.
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 24)]
    if train_data.empty:
        print("No training data found for GW 1-24 in merged_gw_cleaned.csv.")
        return
    print(f"Training data shape (GW 1-24): {train_data.shape}")

    # 3. Select features and target for assists prediction.
    # Base features that we expect to drive assists:
    # - minutes: Playing time, which influences assist opportunities.
    # - expected_assists: A pre-calculated metric based on chance creation.
    # - creativity: A key indicator for a player's ability to create chances.
    # - bps: Bonus points which reflect overall match impact.
    # - transfers_in: Proxy for current form/popularity.
    # - xP: Expected points as an overall performance indicator.
    # - total_points: Overall fantasy performance.
    # - influence: General impact on matches.
    # - expected_goal_involvements: Involvement in scoring opportunities.
    assists_features = [
        "minutes",
        "expected_assists",
        "creativity",
        "bps",
        "transfers_in",
        "xP",
        "total_points",
        "influence",
        "expected_goal_involvements"
    ]
    
    # Check if all features exist in the dataset
    missing_features = [f for f in assists_features if f not in train_data.columns]
    if missing_features:
        print(f"Warning: These features are missing from dataset: {missing_features}")
        # Remove missing features from the list
        assists_features = [f for f in assists_features if f in train_data.columns]
        print(f"Using these features instead: {assists_features}")
    
    target_assists = "assists"

    # Drop rows with missing values in the selected features and target.
    train_data = train_data.dropna(subset=assists_features + [target_assists])
    for col in assists_features + [target_assists]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 4. One-Hot Encode the 'position' column if it exists
    if "position" in train_data.columns:
        train_data = pd.get_dummies(train_data, columns=["position"])
        position_cols = [col for col in train_data.columns if col.startswith("position_")]
        assists_features.extend(position_cols)
        print("Position column found and one-hot encoded.")
    else:
        print("Position column not found. Continuing without position encoding.")

    # Final feature set includes the base features plus the one-hot encoded position columns (if any).
    final_features_assists = assists_features
    print("Final features for assists model:", final_features_assists)

    # Prepare the feature matrix and target vector.
    X_assists = train_data[final_features_assists]
    y_assists = train_data[target_assists]
    print("Sample of prepared features:")
    print(X_assists.head())

    # 5. Train-Test Split and Model Training.
    X_train, X_test, y_train, y_test = train_test_split(X_assists, y_assists, test_size=0.2, random_state=42)
    model_assists = RandomForestRegressor(n_estimators=100, random_state=42)
    model_assists.fit(X_train, y_train)

    # 6. Evaluate the model on the test set.
    predictions_test_assists = model_assists.predict(X_test)
    mae_assists = mean_absolute_error(y_test, predictions_test_assists)
    print("Mean Absolute Error (MAE) on the test set (assists):", mae_assists)

    # 7. Prepare GW 25 data for prediction.
    # Group by player and team, averaging all features to simulate a single representative entry per player.
    gw25_data_assists = train_data.groupby(["name", "team"])[final_features_assists].mean().reset_index()
    gw25_data_assists["GW"] = 25  # Set the gameweek to 25.

    # Use the full final feature set for prediction.
    X_future_assists = gw25_data_assists[final_features_assists]

    # 8. Predict assists for GW 25.
    gw25_data_assists["predicted_assists"] = model_assists.predict(X_future_assists)
    
    # Ensure predictions are non-negative
    gw25_data_assists["predicted_assists"] = np.clip(gw25_data_assists["predicted_assists"], 0, None)

    # 9. Sort the predictions by predicted_assists in descending order.
    gw25_data_assists_sorted = gw25_data_assists.sort_values(by="predicted_assists", ascending=False)

    # 10. Save the sorted predictions to a CSV file.
    out_path = "GW_highest_Predicted_assists.csv"
    gw25_data_assists_sorted.to_csv(out_path, index=False)
    print(f"Sorted predictions (by predicted_assists descending) for GW 25 saved to {out_path}")
    print("Sample predictions:\n", gw25_data_assists_sorted[["name", "team", "GW", "predicted_assists"]].head(10))

if __name__ == "__main__":
    main()