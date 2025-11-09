# predict_live.py
"""
Use the trained Ridge model to predict future AQI (next 1–3 hours or days)
"""

import pandas as pd
import joblib
import requests
from datetime import datetime, timezone

API_KEY = "bfa255ff3598c5ff7f4573729be38b2d"
LAT, LON = 24.8607, 67.0011  # Karachi
MODEL_PATH = "models/ridge_model.joblib"

def fetch_current_data():
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    aqi_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

    w = requests.get(weather_url).json()
    a = requests.get(aqi_url).json()["list"][0]

    data = {
        "timestamp": datetime.now(timezone.utc),
        "temp": w["main"]["temp"],
        "feels_like": w["main"]["feels_like"],
        "humidity": w["main"]["humidity"],
        "pressure": w["main"]["pressure"],
        "wind_speed": w["wind"]["speed"],
        "aqi_reported": a["main"]["aqi"],
        "pm2_5": a["components"]["pm2_5"],
        "pm10": a["components"]["pm10"],
        "no2": a["components"]["no2"],
        "so2": a["components"]["so2"],
        "co": a["components"]["co"],
        "o3": a["components"]["o3"]
    }
    return data

def main():
    model = joblib.load(MODEL_PATH)
    print("✅ Loaded model successfully")

    data = fetch_current_data()
    df = pd.DataFrame([data])

    # drop non-feature columns
    X = df.drop(columns=["timestamp", "aqi_reported"], errors="ignore")

    pred = model.predict(X)[0]
    print(f"🌍 Current AQI (reported): {data['aqi_reported']}")
    print(f"🤖 Predicted next AQI: {pred:.2f}")

if __name__ == "__main__":
    main()
