import requests
from datetime import datetime, timezone

def get_weather_with_forecast(city: str) -> dict:
    """
    Fetch current weather + next 12 hours forecast starting from current hour
    """
    try:
        # Get city coordinates
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url, timeout=10).json()
        #print(geo_response)

        if "results" not in geo_response or len(geo_response["results"]) == 0:
            return {"error": f"City '{city}' not found."}

        lat = geo_response["results"][0]["latitude"]
        lon = geo_response["results"][0]["longitude"]
        city_name = geo_response["results"][0]["name"]

        # Get current weather + hourly forecast
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            f"&hourly=temperature_2m,precipitation_probability,weather_code"
            f"&forecast_days=2"
            f"&timezone=auto"
        )
        weather_response = requests.get(weather_url, timeout=10).json()

        current = weather_response.get("current", {})
        hourly = weather_response.get("hourly", {})

        # Get current time
        current_time = current.get("time")  # Example: "2026-06-22T07:00"

        # Find the index of current hour in hourly data
        try:
            current_index = hourly["time"].index(current_time)
        except ValueError:
            current_index = 0  # fallback

        # Get next 12 hours starting from current hour
        forecast_list = []
        for i in range(current_index, min(current_index + 24, len(hourly["time"]))):
            forecast_list.append({
                "time": hourly["time"][i],
                "temperature": hourly["temperature_2m"][i],
                "rain_probability": hourly["precipitation_probability"][i],
                "weather_code": hourly["weather_code"][i]
            })

        weather_data = {
            "city": city_name,
            "current": {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "time": current.get("time")
            },
            "forecast": forecast_list
        }

        return weather_data

    except Exception as e:
        return {"error": str(e)}