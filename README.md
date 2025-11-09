# Pearls AQI Predictor 🌿

Predict the Air Quality Index (AQI) for the next 3 days using a fully automated ML pipeline, from data collection to a real-time dashboard.

---

## 🎯 Project Goal

Build an end-to-end system that:

- Collects weather and pollutant data from APIs
- Trains ML models to forecast AQI
- Provides real-time predictions via a web dashboard
- Automates the entire workflow using CI/CD pipelines

---

## 🧩 Main Components

### 1. Feature Pipeline
- Fetch raw data from APIs like **AQICN** and **OpenWeather**
- Extract important features: `PM2.5, PM10, NO₂, SO₂, CO, O₃, temp, humidity, wind_speed, date/time`
- Compute derived features:
  - Hour, Day, Month
  - Lag features (previous AQI)
  - AQI change rate
- Store processed features in a **Feature Store** (CSV or database)

### 2. Training Pipeline
- Fetch historical data from Feature Store
- Split into **train/test sets**
- Train multiple ML models:
  - Random Forest
  - Ridge Regression
  - Optional: LSTM/GRU for time-series
- Evaluate using **RMSE, MAE, R²**
- Store the best model in a **Model Registry**

### 3. Automation Pipeline
- Automate tasks using **Apache Airflow** or **GitHub Actions**
- Example schedule:
  - Feature pipeline → hourly
  - Training pipeline → daily

### 4. Web Application (Dashboard)
- Display **current AQI** and **predicted AQI** for the next 3 days
- Visualizations:
  - Time series plots
  - Pollutant contribution
  - Alerts for hazardous AQI levels (>150)
- Built with **Streamlit** or **Flask/FastAPI**

---

## 📊 Bonus Features
- Feature Importance with **SHAP** or **LIME**
- Correlation heatmaps
- Actual vs predicted AQI plots
- Trend line for upcoming days

---

## 🛠️ Technology Stack
- Python
- Scikit-learn / TensorFlow
- Apache Airflow / GitHub Actions
- Streamlit / Flask / FastAPI
- AQICN / OpenWeather APIs
- SHAP / LIME
- Git

---

## 📂 Project Structure
