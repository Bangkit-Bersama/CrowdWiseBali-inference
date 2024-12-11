import datetime
import json
import os

import numpy as np
import pandas as pd
from keras.api.models import load_model
from sklearn.preprocessing import MinMaxScaler

place_ids: dict[str, str]

df1 = pd.read_csv("combined_data2.csv")

df1["Year"] = pd.to_datetime(df1["Date"]).dt.year
df1["YearMonth"] = pd.to_datetime(df1["Date"]).dt.to_period("M").astype(str)
df1 = df1[["Title", "Date", "Day", "Hour", "OccupancyPercent", "Year", "YearMonth"]]


def inference(place_id: str, date: str, hour: int) -> tuple[any, str]:
    location_name = place_ids.get(place_id)
    if location_name == None:
        return (None, f"Place ID '{place_id}' not registered in database.")

    location_data = df1[df1["Title"] == location_name]
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

    # Load model
    model_path = f"models/{place_id}.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
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


def main():
    global place_ids

    with open("places.json", "rt") as f:
        data = json.loads(f.read())
        place_ids = data["placeIds"]

    print(inference("ChIJKWD5VBdH0i0ReivZYCJIm24", "2024-01-01", 17))


if __name__ == "__main__":
    main()
