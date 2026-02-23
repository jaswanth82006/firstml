from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load model files
model = joblib.load("bus_model.pkl")
le_route = joblib.load("route_encoder.pkl")
le_stop = joblib.load("stop_encoder.pkl")
le_weather = joblib.load("weather_encoder.pkl")

FEATURE_ORDER = ['hour', 'route', 'stop', 'weather', 'weekday']
TOTAL_SEATS = 30

@app.route("/")
def home():
    return "Bus Occupancy Prediction API is running"

@app.route("/predict", methods=["GET"])
def predict():
    try:
        route = request.args.get("route")
        stop = request.args.get("stop")
        hour = int(request.args.get("hour"))
        weekday = int(request.args.get("weekday"))
        weather = request.args.get("weather")

        route_enc = le_route.transform([route])[0]
        stop_enc = le_stop.transform([stop])[0]
        weather_enc = le_weather.transform([weather])[0]

        sample = pd.DataFrame([[hour, route_enc, stop_enc, weather_enc, weekday]],
                              columns=FEATURE_ORDER)

        predicted_passengers = int(model.predict(sample)[0])
        overcrowded = predicted_passengers > TOTAL_SEATS

        reasons = []
        if 16 <= hour <= 19:
            reasons.append("peak office hour")

        reasons.append("working day" if weekday == 1 else "weekend")
        reasons.append("rainy weather" if weather == "rain" else "normal weather")

        return jsonify({
            "predicted_passengers": predicted_passengers,
            "total_seats": TOTAL_SEATS,
            "status": "Overcrowded" if overcrowded else "Not Overcrowded",
            "explanation": "Because of " + ", ".join(reasons)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
