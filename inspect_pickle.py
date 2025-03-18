import pickle
import pandas as pd

# Load the trained model
with open("minutes_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the dataset used for training (ensure this is the correct file)
data = pd.read_csv("data/cleaned_merged_seasons.csv")

# Define feature names (you need to use the same features you trained the model with)
features = [
    "feature_1", "feature_2", "feature_3",  # Replace these with actual feature names
]

# Ensure features are correct
if len(features) != len(model.feature_importances_):
    print("Error: Mismatch between feature list and model importances")
else:
    # Print feature importances with names
    importance_dict = dict(zip(features, model.feature_importances_))
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
