# predict_3day_aqi.py
"""
3-Day AQI Forecast Script (fixed)
- Ensures feature names match training (adds pollutant columns to forecast rows)
- Predicts next 72 hours and prints 3-day averages
"""

import os
import joblib
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta

# ==== CONFIG ====
API_KEY = "bfa255ff3598c5ff7f4573729be38b2d"
LAT, LON = 24.8607, 67.0011  # Karachi
MODEL_PATH = "models/ridge_model.joblib"
FORECAST_CSV = "forecast_72h.csv"

# ==== HELPERS ====
def fetch_forecast(lat, lon):
    """Fetch 5-day / 3-hour weather forecast from OpenWeather and limit to next 72 hours"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    r = requests.get(url); r.raise_for_status()
    data = r.json()

    records = []
    for entry in data["list"]:
        ts = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        records.append({
            "timestamp": ts,
            "temp": entry["main"]["temp"],
            "feels_like": entry["main"]["feels_like"],
            "humidity": entry["main"]["humidity"],
            "pressure": entry["main"]["pressure"],
            "wind_speed": entry["wind"]["speed"],
        })

    df = pd.DataFrame(records)
    now = datetime.now(timezone.utc)
    limit_time = now + timedelta(hours=72)
    df = df[df["timestamp"] <= limit_time].reset_index(drop=True)
    return df

def fetch_current_pollution(lat, lon):
    """Fetch latest pollution / components from OpenWeather"""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    r = requests.get(url); r.raise_for_status()
    a = r.json()["list"][0]
    comps = a["components"]
    return {
        "aqi_reported": a["main"]["aqi"],
        "pm2_5": comps.get("pm2_5", None),
        "pm10": comps.get("pm10", None),
        "no2": comps.get("no2", None),
        "so2": comps.get("so2", None),
        "co": comps.get("co", None),
        "o3": comps.get("o3", None)
    }

# ==== MAIN ====
def main():
    print("🔹 Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("🌦️ Fetching forecast data for next 72 hours...")
    weather_df = fetch_forecast(LAT, LON)
    print(f"✅ Collected {len(weather_df)} forecast records (next 3 days)")

    print("🟠 Fetching latest pollution components...")
    pollution = fetch_current_pollution(LAT, LON)
    # add pollutant columns to each forecast row (matching training features)
    for k, v in pollution.items():
        # training used 'aqi_reported' as column but then dropped it; model used pollutant columns
        weather_df[k] = v

    # Prepare features: drop timestamp and any extra columns not used during training
    # Determine expected feature names from model (if pipeline, get named step input)
    # We'll infer expected features from training pipeline by loading model and checking .named_steps if exists
    expected_features = None
    try:
        # model might be a Pipeline with final estimator; find the preprocessing transformer to inspect feature names is complex.
        # Simpler: we know training used these columns (11 features): 
        expected_features = [
            "temp","feels_like","humidity","pressure","wind_speed",
            "pm2_5","pm10","no2","so2","co","o3"
        ]
    except Exception:
        expected_features = [
            "temp","feels_like","humidity","pressure","wind_speed",
            "pm2_5","pm10","no2","so2","co","o3"
        ]

    # Ensure all expected features exist in weather_df (create if missing with NaN)
    for feat in expected_features:
        if feat not in weather_df.columns:
            weather_df[feat] = pd.NA

    # Build X with exact column order
    X = weather_df[expected_features].copy()

    print("🤖 Predicting next 72 hours AQI...")
    preds = model.predict(X)
    weather_df["predicted_aqi"] = preds

    # Save detailed hourly forecast
    weather_df.to_csv(FORECAST_CSV, index=False)

    # Group by date and show 3-day average forecast (only first 3 unique dates)
    weather_df["date"] = weather_df["timestamp"].dt.date
    daily = weather_df.groupby("date")["predicted_aqi"].mean().reset_index()
    daily = daily.head(3)

    print("\n📅 3-Day AQI Forecast (Average by Day):")
    print(daily.to_string(index=False))
    print(f"\n✅ Saved detailed forecast to {FORECAST_CSV}")

if __name__ == "__main__":
    main()
