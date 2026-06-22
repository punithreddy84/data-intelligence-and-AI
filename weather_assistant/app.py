import streamlit as st
from groq import Groq
from weather_api import get_weather_with_forecast
from prompt_builder import build_weather_prompt

st.set_page_config(page_title="AI Weather Chatbot", page_icon="🌤️")
st.title("🌤️ AI Weather Chatbot (Groq + Dynamic Context Injection)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "weather_data" not in st.session_state:
    st.session_state.weather_data = None

if "city" not in st.session_state:
    st.session_state.city = ""

if "questions_left" not in st.session_state:
    st.session_state.questions_left = 5

# Sidebar
with st.sidebar:
    st.header("Settings")

    groq_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    if st.button("Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.weather_data = None
        st.session_state.questions_left = 5
        st.rerun()

# Main UI
if st.session_state.questions_left > 0:

    if not st.session_state.weather_data:

        city = st.text_input(
            "Enter City Name",
            value="Hyderabad"
        )

        if st.button("Get Weather & Start Chat"):

            if not groq_key:
                st.error("Please enter Groq API Key")

            else:
                with st.spinner("Fetching weather..."):
                    weather = get_weather_with_forecast(city)

                if "error" in weather:
                    st.error(weather["error"])

                else:
                    st.session_state.weather_data = weather
                    st.session_state.city = city
                    st.rerun()

    else:

        st.success(
            f"Chatting about: **{st.session_state.city}** | "
            f"Questions left: {st.session_state.questions_left}"
        )

        user_question = st.text_input(
            "Ask about the weather:"
        )

        if st.button("Send") and user_question:

            prompt = build_weather_prompt(
                st.session_state.weather_data,
                user_question,
                st.session_state.chat_history
            )

            try:
                client = Groq(api_key=groq_key)

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                answer = response.choices[0].message.content

                st.session_state.chat_history.append(
                    (user_question, answer)
                )

                st.session_state.questions_left -= 1

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

        # Show Chat History
        for q, a in st.session_state.chat_history:
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)

else:

    st.warning(
        "You have reached the 5-question limit for this city."
    )

    if st.button("Start New City"):
        st.session_state.chat_history = []
        st.session_state.weather_data = None
        st.session_state.questions_left = 5
        st.rerun()