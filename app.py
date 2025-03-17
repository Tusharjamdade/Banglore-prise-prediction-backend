from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pickle
import numpy as np
import os

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Global Variables
__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    """Loads ML model and column names from artifacts folder"""
    global __data_columns
    global __locations
    global __model

    try:
        # Get absolute path (Handle Render File Path Issues)
        artifact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        print(f"🔍 Checking artifacts path: {artifact_path}")

        # Load column names
        columns_file = os.path.join(artifact_path, "columns.json")
        if not os.path.exists(columns_file):
            raise FileNotFoundError(f"❌ columns.json not found at {columns_file}")

        print(f"📂 Loading columns from: {columns_file}")
        with open(columns_file, "r") as f:
            data = json.load(f)
            if "data_columns" not in data:
                raise ValueError("❌ columns.json does not contain 'data_columns' key")

            __data_columns = data["data_columns"]
            __locations = __data_columns[3:]  # First 3 columns are sqft, bath, bhk
            print(f"✅ Loaded columns: {__data_columns}")

        # Load trained model
        model_file = os.path.join(artifact_path, "banglore_home_prices_model.pickle")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"❌ Model file not found at {model_file}")

        print(f"📂 Loading model from: {model_file}")
        with open(model_file, "rb") as f:
            __model = pickle.load(f)
            print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# Get List of Locations
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """Returns a list of all available locations"""
    if __locations is None:
        return jsonify({"error": "Locations not loaded"}), 500
    return jsonify({"locations": __locations})

# Predict Home Price
@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """Predicts home price based on user input"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input, expected JSON"}), 400

        # Extract input values
        total_sqft = float(data.get('total_sqft', 0))
        location = data.get('location', "").strip()
        bhk = int(data.get('bhk', 0))
        bath = int(data.get('bath', 0))

        # Validate inputs
        if not location or total_sqft <= 0 or bhk <= 0 or bath <= 0:
            return jsonify({"error": "Missing or invalid parameters"}), 400

        # Get estimated price
        estimated_price = get_estimated_price(location, total_sqft, bhk, bath)
        if isinstance(estimated_price, dict):  # If an error dictionary is returned
            return jsonify(estimated_price), 500

        return jsonify({"estimated_price": estimated_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home Price Prediction Logic
def get_estimated_price(location, sqft, bhk, bath):
    """Returns estimated price based on input features"""
    try:
        if __data_columns is None or __model is None:
            return {"error": "Model or columns not loaded properly"}

        loc_index = __data_columns.index(location.lower()) if location.lower() in __data_columns else -1

        # Create input feature array
        x = np.zeros(len(__data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1  # Mark the correct location

        # Predict price
        return round(__model.predict([x])[0], 2)

    except Exception as e:
        return {"error": str(e)}

# Start Flask Server
if __name__ == "__main__":
    print("🚀 Starting Flask Server for Home Price Prediction...")
    load_saved_artifacts()  # Load model and columns
    app.run(host="0.0.0.0", port=5000, debug=True)  # Run Flask app
