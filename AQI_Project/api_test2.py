import requests

API_KEY = "bfa255ff3598c5ff7f4573729be38b2d"
lat, lon = 24.8607, 67.0011

# Pollution Data
pollution_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
pollution = requests.get(pollution_url)
print("Pollution API:", pollution.status_code, pollution.text[:200])

# Weather Data
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
weather = requests.get(weather_url)
print("Weather API:", weather.status_code, weather.text[:200])
