import pickle

# Replace 'minutes_rf_model.pkl' with the path to your pickle file if necessary
with open('minutes_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Print out some information about the loaded object
print("Loaded object type:", type(model))
print(model)
