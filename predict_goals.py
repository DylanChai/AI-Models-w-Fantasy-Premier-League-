import pickle
import numpy as np

# Load the trained model and feature list
with open("goals_rf_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features"]

# Print expected features
print("Model loaded successfully!")
print("Model expects", len(features), "features.")
print("Expected feature order:", features)

# ---------------------------
# Example Player Input
# ---------------------------
# Example of a new player:
# - total_points: 120
# - assists: 10
# - minutes: 2500
# - selected: 50000
# - bps: 200
# - threat: 80
# - creativity: 60
# - influence: 50
# - transfers_in: 3000
# 
# Team Strength:
# - team_points: 40 (Team has 40 league points so far)
# - team_goals_scored: 55 (Team has scored 55 goals so far)
# - team_goals_diff: 20 (Team's goal difference is +20)
#
# Position Encoding:
# - position_DEF: 0, position_MID: 0, position_FWD: 1

new_player = np.array([[120, 10, 2500, 50000, 200, 80, 60, 50, 3000,  
                         40, 55, 20, 0, 0, 1]])  # 15 features

# Ensure input shape matches model expectation
print("New player input shape:", new_player.shape)

# Make prediction
predicted_goals = model.predict(new_player)

# Print result
print(f"Predicted Goals Scored: {predicted_goals[0]}")
