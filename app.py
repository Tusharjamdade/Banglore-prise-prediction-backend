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

# Load Artifacts (Model + Columns)
def load_saved_artifacts():
    """Loads ML model and column names from artifacts folder"""
    global __data_columns
    global __locations
    global __model

    try:
        artifact_path = os.path.join(os.path.dirname(__file__), "artifacts")

        # Load column names
        with open(os.path.join(artifact_path, "columns.json"), "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]  # First 3 columns are sqft, bath, bhk

        # Load trained model
        with open(os.path.join(artifact_path, "banglore_home_prices_model.pickle"), "rb") as f:
            __model = pickle.load(f)

        print("✅ Model and artifacts loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# Get List of Locations
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """Returns a list of all available locations"""
    if not __locations:
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
        return jsonify({"estimated_price": estimated_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Home Price Prediction Logic
def get_estimated_price(location, sqft, bhk, bath):
    """Returns estimated price based on input features"""
    try:
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
    app.run(debug=True)  # Run Flask app
