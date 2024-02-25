from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_locations')
def get_locations():
    response = jsonify({
        'locations': util.get_all_locations()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask server for House Price Prediction...")
    app.run()