import datetime
import json
import os

import flask
import flask_cors

from keras.src.models.sequential import Sequential
from keras.api.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MODEL_VERSION = 1

app = flask.Flask(__name__)
flask_cors.CORS(app)

place_ids: dict[str, str]

tdata: pd.DataFrame

loaded_models: dict[str, Sequential] = {}


def inference(place_id: str, date: str, hour: int) -> tuple[any, str]:
    global loaded_models

    location_name = place_ids.get(place_id)
    if location_name == None:
        return (None, f"Place ID '{place_id}' not registered in database.")

    location_data = tdata[tdata["Title"] == location_name]
    if location_data.empty:
        return (None, f"Location '{location_name}' not found in the dataset.")

    # Normalize data for scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    location_data_scaled = scaler.fit_transform(location_data[["OccupancyPercent"]])

    # Prepare data for LSTM
    window_size = 24
    X, y = [], []
    for i in range(window_size, len(location_data_scaled)):
        X.append(location_data_scaled[i - window_size : i, 0])
        y.append(location_data_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model: Sequential = loaded_models.get(place_id)
    if model == None:
        # Load model
        model_path = f"models/{MODEL_VERSION}/{place_id}.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            loaded_models[place_id] = model
        else:
            return (None, "Model for the location not found.")

    # Predict occupancy percent
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    predictions = np.clip(predictions, 0, 100)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    y_actual = np.clip(y_actual, 0, 100)

    # Convert date string to datetime
    user_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    last_date = pd.to_datetime(location_data["Date"].max())

    # Convert hour to integer
    hour = int(hour)  # Ensure hour is an integer

    if user_date < last_date:
        delta = last_date - user_date
        total_hours = delta.days * 24 + (hour - last_date.hour)
    else:
        total_hours = (user_date - last_date).days * 24 + hour

    if total_hours < 0 or total_hours >= len(predictions):
        return (
            None,
            f"Date and hour must be within the predicted range. Available range: {last_date} to {last_date + datetime.timedelta(hours=len(predictions))}.",
        )

    if not (9 <= hour <= 22):
        return (0, None)

    predicted_value = predictions[total_hours - 1][0]
    predicted_value = np.clip(predicted_value, 0, 100)
    return (predicted_value, None)


# Route to handle predictions (POST request)
@app.route("/", methods=["POST"])
def predict():
    # Get data from JSON body
    data = flask.request.get_json()

    place_id = data.get("placeId")
    date = data.get("date")
    hour = data.get("hour")

    if not place_id or not date or not hour:
        return flask.jsonify({"error": "Missing required parameters"}), 400

    result = inference(place_id, date, hour)

    if result[0] == None:
        return flask.jsonify({"error": result[1]})
    
    occupancy = float(result[0])

    # Return response
    return flask.jsonify({"prediction": occupancy})


def main():
    global place_ids
    global tdata

    with open(f"models/{MODEL_VERSION}/places.json", "rt") as f:
        data = json.loads(f.read())
        place_ids = data["placeIds"]

    tdata_raw = pd.read_csv(f"models/{MODEL_VERSION}/tdata.csv")
    tdata_raw["Year"] = pd.to_datetime(tdata_raw["Date"]).dt.year
    tdata_raw["YearMonth"] = (
        pd.to_datetime(tdata_raw["Date"]).dt.to_period("M").astype(str)
    )

    tdata = tdata_raw[
        ["Title", "Date", "Day", "Hour", "OccupancyPercent", "Year", "YearMonth"]
    ]

    app.run(host='0.0.0.0', port=3000)


if __name__ == "__main__":
    main()
