# feature_pipeline.py
"""
Feature Pipeline for AQI Project (10Pearls Shine)
------------------------------------------------
- Fetches both current and historical (past 7 days) AQI + weather data
- Combines results into features.csv for model training
- Uses OpenWeather APIs (Pollution + Weather)
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# ==== CONFIGURATION ====
API_KEY = "bfa255ff3598c5ff7f4573729be38b2d"   # <-- Your API key added here
LAT, LON = 24.8607, 67.0011  # Karachi coordinates
OUT_CSV = "features.csv"

# ==== FUNCTIONS ====
def fetch_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    return {
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
    }

def fetch_aqi(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()["list"][0]
    comps = data["components"]
    return {
        "aqi_reported": data["main"]["aqi"],
        "pm2_5": comps["pm2_5"],
        "pm10": comps["pm10"],
        "no2": comps["no2"],
        "so2": comps["so2"],
        "co": comps["co"],
        "o3": comps["o3"],
    }

def fetch_historical(lat, lon, timestamp):
    """Fetch historical AQI + weather using OpenWeather history API"""
    url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={timestamp}&end={timestamp + 3600}&appid={API_KEY}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            if "list" in data and len(data["list"]) > 0:
                comps = data["list"][0]["components"]
                return {
                    "aqi_reported": data["list"][0]["main"]["aqi"],
                    "pm2_5": comps["pm2_5"],
                    "pm10": comps["pm10"],
                    "no2": comps["no2"],
                    "so2": comps["so2"],
                    "co": comps["co"],
                    "o3": comps["o3"],
                }
    except Exception as e:
        print("Historical fetch error:", e)
    return None

def collect_data():
    records = []

    # ---- 1️⃣ Current data (for reference) ----
    try:
        w = fetch_weather(LAT, LON)
        a = fetch_aqi(LAT, LON)
        row = {"timestamp": datetime.now(timezone.utc)}
        row.update(w)
        row.update(a)
        records.append(row)
        print("[LIVE] Collected current AQI + weather")
    except Exception as e:
        print("Live fetch error:", e)

    # ---- 2️⃣ Historical backfill for past 7 days (hourly) ----
    print("Fetching past 7 days historical data...")
    now = datetime.now(timezone.utc)
    for days_ago in range(1, 8):  # last 7 days
        day_time = now - timedelta(days=days_ago)
        for hour in range(0, 24):  # each hour
            ts = int((day_time.replace(hour=hour, minute=0, second=0)).timestamp())
            hist = fetch_historical(LAT, LON, ts)
            if hist:
                row = {"timestamp": datetime.fromtimestamp(ts, tz=timezone.utc)}
                row.update(hist)
                records.append(row)
    print(f"✅ Collected {len(records)} total records")

    # ---- 3️⃣ Save to CSV ----
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved data to {OUT_CSV} ({len(df)} rows)")
    return df

# ==== MAIN ====
if __name__ == "__main__":
    if not API_KEY:
        print("❌ Please set your OPENWEATHER_API_KEY environment variable first!")
    else:
        collect_data()
