def build_weather_prompt(weather_data: dict, user_question: str, chat_history: list = None) -> str:
    """
    Build dynamic prompt with current weather + forecast + chat history
    """
    if "error" in weather_data:
        return f"Error: {weather_data['error']}"

    current = weather_data["current"]
    forecast = weather_data["forecast"]

    # Format forecast nicely
    forecast_text = "\n".join([
        f"- {f['time']}: {f['temperature']}°C, Rain chance: {f['rain_probability']}%"
        for f in forecast[:8]   # Show next 8 hours
    ])

    history_text = ""
    if chat_history:
        history_text = "\n### Previous Questions & Answers:\n"
        for q, a in chat_history:
            history_text += f"User: {q}\nAssistant: {a}\n\n"

    prompt = f"""You are a helpful Weather Assistant.

### Current Weather in {weather_data['city']}:
- Temperature: {current['temperature']}°C
- Humidity: {current['humidity']}%
- Wind Speed: {current['wind_speed']} km/h

### Today's Forecast (Next few hours):
{forecast_text}

{history_text}
### New User Question:
{user_question}

Instructions:
- Use only the weather data and forecast provided above.
- Answer naturally and helpfully.
- If the user asks something unrelated to weather, politely say you can only help with weather.
"""
    return prompt