# streamlit_app.py
"""
Run: streamlit run streamlit_app.py
Displays: current features if available, next-3-days predictions, simple charts, alert banner.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

MODEL_PATH = "models/rf_model.joblib"  # adjust to saved model filename
FEATURES_CSV = "features.csv"

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")
st.title("Pearls AQI Predictor — Next 3 Days Forecast")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    st.sidebar.success("Model loaded: " + MODEL_PATH)
except Exception as e:
    st.sidebar.error("Model not found. Run training pipeline first.")
    st.stop()

# Load latest features
df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
st.sidebar.write("Data last updated:", df["timestamp"].max())

st.header("Current / Recent AQI")
latest = df.sort_values("timestamp").iloc[-1]
st.metric(label="Latest reported AQI", value=float(latest.get("aqi_reported", np.nan)))

# Build inputs for next 3 days (example: keep last row and shift timestamps)
h = 24  # horizon per day
X_latest = df.drop(columns=["timestamp", "aqi_reported"], errors="ignore").iloc[-1:]
preds = []
for day in range(1,4):
    # this simple approach uses the same features but rolled: in production compute forecast features properly
    pred = model.predict(X_latest)[0]
    preds.append(pred)

# Display predictions
st.subheader("Predicted AQI — Next 3 Days")
days = [(datetime.utcnow() + timedelta(days=i)).date().isoformat() for i in range(1,4)]
pred_df = pd.DataFrame({"date": days, "pred_aqi": preds})
st.table(pred_df)

# Alert if any hazardous
hazardous = pred_df[pred_df["pred_aqi"] >= 151]
if not hazardous.empty:
    st.error(f"Hazardous AQI predicted on: {', '.join(hazardous['date'].astype(str))}")

# Plot
fig, ax = plt.subplots()
ax.plot(pred_df["date"], pred_df["pred_aqi"], marker="o")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted AQI")
ax.set_title("3-day AQI Forecast")
st.pyplot(fig)
