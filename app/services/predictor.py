import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime, timedelta

# Load model and scaler
model = tf.keras.models.load_model("app/models/glucose_lstm_model.h5")
scaler = joblib.load("app/models/glucose_scaler.gz")

WINDOW_SIZE = 5  # Adjust this to match your training config

def predict_glucose(values: list[float]):
    if len(values) < WINDOW_SIZE:
        return {"error": f"Need at least {WINDOW_SIZE} values for prediction"}

    # Take last WINDOW_SIZE values
    input_seq = np.array(values[-WINDOW_SIZE:]).reshape(-1, 1)
    input_scaled = scaler.transform(input_seq).reshape(1, WINDOW_SIZE, 1)

    # Predict next 3 values
    pred = model.predict(input_scaled)[0]
    forecast = scaler.inverse_transform(pred.reshape(-1, 1)).flatten().tolist()

    # Create timestamped forecast
    now = datetime.utcnow()
    result = [
        {"timestamp": (now + timedelta(hours=i+1)).isoformat(), "value": round(val, 2)}
        for i, val in enumerate(forecast)
    ]
    return {"predictions": result}
