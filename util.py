import pickle
import json
import numpy as np
import os

# Global variables
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """Returns estimated price based on input features"""
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1  # Location not found

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1  # Mark the correct location

    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    """Loads model and column information"""
    print("Loading saved artifacts...")

    global __data_columns
    global __locations
    global __model

    # Load column names
    artifact_path = os.path.join(os.path.dirname(__file__), "artifacts")
    
    with open(os.path.join(artifact_path, "columns.json"), "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # First 3 columns are sqft, bath, bhk

    # Load trained model
    with open(os.path.join(artifact_path, "banglore_home_prices_model.pickle"), "rb") as f:
        __model = pickle.load(f)

    print("Loading saved artifacts...done")

def get_location_names():
    """Returns a list of all locations"""
    return __locations

def get_data_columns():
    """Returns all column names"""
    return __data_columns

# Debugging: Load and test functions when running directly
if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Indiranagar', 1000, 3, 3))
    print(get_estimated_price('Whitefield', 1200, 2, 2))
