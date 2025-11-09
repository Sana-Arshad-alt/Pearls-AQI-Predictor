import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="3-Day AQI Forecast", page_icon="🌤️", layout="centered")

# --- Title
st.title("🌤️ 3-Day Air Quality Index (AQI) Forecast")
st.markdown("This dashboard shows the predicted AQI for the next 3 days based on weather and pollution data.")

# --- Load data
try:
    df = pd.read_csv("forecast_72h.csv")
except FileNotFoundError:
    st.error("⚠️ forecast_72h.csv not found! Please run predict_3day_aqi.py first.")
    st.stop()

# --- Convert timestamp to date if needed
if "timestamp" in df.columns:
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
elif "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"]).dt.date

# --- Average by date
daily = df.groupby("date")["predicted_aqi"].mean().reset_index()

# --- AQI color scale
def get_aqi_color(aqi):
    if aqi <= 50:
        return "green"
    elif aqi <= 100:
        return "yellow"
    elif aqi <= 150:
        return "orange"
    elif aqi <= 200:
        return "red"
    else:
        return "purple"

daily["color"] = daily["predicted_aqi"].apply(get_aqi_color)

# --- Plot chart
fig = px.bar(
    daily,
    x="date",
    y="predicted_aqi",
    color="color",
    color_discrete_map="identity",
    title="Predicted AQI (Next 3 Days)",
    labels={"predicted_aqi": "Predicted AQI", "date": "Date"},
)
fig.update_layout(yaxis=dict(range=[0, max(daily["predicted_aqi"]) + 10]))

st.plotly_chart(fig, use_container_width=True)

# --- Table
st.subheader("📋 Forecast Data")
st.dataframe(daily[["date", "predicted_aqi"]].style.format({"predicted_aqi": "{:.2f}"}))

st.markdown("---")
st.caption("Developed by Sana Arshad 🌿 | Powered by Machine Learning & OpenWeather API")
