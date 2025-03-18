import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

def main():
    # 1. Load the merged file
    data_path = "data/2024-25/gws/merged_gw.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    try:
        data = pd.read_csv(data_path, low_memory=False, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return

    # 2. Use GW 1-24 for training
    train_data = data[(data["GW"] >= 1) & (data["GW"] <= 24)]
    if train_data.empty:
        print("No training data found for GW 1-24 in merged_gw.csv.")
        return

    # 3. Select features & target
    features = [
        "minutes",
        "xP",
        "influence",
        "ict_index",
        "threat",
        "creativity",
        "bps",
        "expected_goal_involvements",
        "transfers_in"
    ]
    target = "total_points"

    # Drop rows with missing values
    train_data = train_data.dropna(subset=features + [target])
    for col in features + [target]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

    # 4. One-Hot Encode the 'position' column
    train_data = pd.get_dummies(train_data, columns=["position"])
    position_cols = [col for col in train_data.columns if col.startswith("position_")]
    final_features = features + position_cols

    # 5. Split training data
    X = train_data[final_features]
    y = train_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. Evaluate the model
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print("Mean Absolute Error (MAE) on the test set:", mae)

    # 8. Prepare GW 25 data for prediction
    gw25_data = train_data.groupby(["name", "team"])[final_features].mean().reset_index()
    gw25_data["GW"] = 25

    # 9. Predict total points for GW 25
    gw25_data["predicted_total_points"] = model.predict(gw25_data[final_features])

    # 10. Save predictions
    out_path = "GW_highest_Predicted_total_points.csv"
    gw25_data.to_csv(out_path, index=False)
    print(f"Predictions for GW 25 saved to {out_path}")

if __name__ == "__main__":
    main()