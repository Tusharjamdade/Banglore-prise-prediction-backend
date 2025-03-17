from flask import Flask, request, jsonify
import util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """Returns a list of all available locations"""
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """Predicts home price based on input JSON"""
    try:
        data = request.get_json()  # Use get_json() for JSON data
        if not data:
            return jsonify({"error": "Invalid input, expected JSON"}), 400

        total_sqft = float(data.get('total_sqft', 0))
        location = data.get('location', "").strip()
        bhk = int(data.get('bhk', 0))
        bath = int(data.get('bath', 0))

        if not location or total_sqft <= 0 or bhk <= 0 or bath <= 0:
            return jsonify({"error": "Missing or invalid parameters"}), 400

        response = jsonify({
            'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask Server for Home Price Prediction...")
    util.load_saved_artifacts()
    app.run(debug=True)  # Enable Debug Mode
