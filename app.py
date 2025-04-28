import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the trained model using joblib
model = joblib.load('house_price_model.pkl')

@app.route("/")
def home():
    return "House Price Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    # Receive input from the request as JSON
    data = request.get_json()  # Accepting the data as JSON input

    # Extract features from the incoming JSON data
    area = data.get('area', [8500])               # Area of the house
    bedrooms = data.get('bedrooms', [5])           # Number of bedrooms
    bathrooms = data.get('bathrooms', [2])         # Number of bathrooms
    stories = data.get('stories', [2])             # Number of stories
    mainroad = data.get('mainroad', [1])           # Mainroad value (1 for Yes)
    guestroom = data.get('guestroom', [1])         # Guestroom value (1 for Yes)
    basement = data.get('basement', [0])           # Basement value (0 for No)
    hotwaterheating = data.get('hotwaterheating', [1])  # Hot water heating value (1 for Yes)
    airconditioning = data.get('airconditioning', [0])  # Air conditioning value (0 for No)
    parking = data.get('parking', [2])             # Number of parking spaces
    prefarea = data.get('prefarea', [1])           # Prefarea value (1 for Yes)
    furnishingstatus = data.get('furnishingstatus', [0.5])  # Furnishing status (e.g., Semi-Furnished)

    # Create feature array from the extracted data
    features = np.array([[
        area, bedrooms, bathrooms, stories, mainroad, guestroom,
        basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus
    ]])

    # Make prediction using the loaded model
    prediction = model.predict(features)[0]  # Make prediction
    newValue=prediction/100
    newprediction = round(newValue, 1)
    return jsonify({"predicted_price": newprediction})

if __name__ == "__main__":
    app.run(debug=True)
