# Importing necessary libraries
from flask import Flask, request, jsonify
import util

# Creating a Flask app instance
app = Flask(__name__)

# Endpoint for getting location names
@app.route('/get_locations', methods=['GET'])
def get_locations():
    # Retrieving location names using the utility function
    response = jsonify({
        'locations': util.get_locations()
    })
    
    # Allowing cross-origin resource sharing (CORS)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

# Endpoint for predicting home price
@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    # Extracting input parameters from the request form
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    # Getting the estimated price using the utility function
    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })

    # Allowing cross-origin resource sharing (CORS)
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

# Starting the Flask server
if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    
    # Loading saved artifacts (model, column names, etc.) using the utility function
    util.load_saved_artifacts()
    
    # Running the Flask app
    app.run()
