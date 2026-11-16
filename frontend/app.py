
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze-farm"

st.set_page_config(
    page_title="Smart Agriculture Advisor",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Smart Agriculture Advisor")
st.subheader("Your AI-powered farming assistant")

st.write(
    "Enter your farm details to receive guidance from "
    "five specialized AI agents."
)

st.divider()

with st.form("farm_details_form"):

    st.header("🌱 Farm Information")

    col1, col2 = st.columns(2)

    with col1:
        location = st.text_input(
            "Farm Location",
            placeholder="Example: Karnataka, India",
        )

        soil_type = st.selectbox(
            "Soil Type",
            ["Loamy", "Clay", "Sandy", "Silty", "Black soil", "Other"],
        )

        soil_ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
        )

        water_availability = st.selectbox(
            "Water Availability",
            ["Low", "Moderate irrigation", "High"],
        )

    with col2:
        season = st.selectbox(
            "Season",
            ["Kharif", "Rabi", "Zaid"],
        )

        intended_crop = st.text_input(
            "Intended Crop",
            placeholder="Example: Maize",
        )

        growth_stage = st.selectbox(
            "Crop Growth Stage",
            ["Unknown", "Germination", "Vegetative stage",
             "Flowering", "Fruiting", "Maturity"],
        )

        weather_information = st.text_area(
            "Weather Information",
            placeholder="Enter available rainfall or weather details",
        )

    symptoms = st.text_area(
        "Pest or Disease Symptoms",
        placeholder="Example: Yellow leaves and small holes",
    )

    submitted = st.form_submit_button(
        "🌾 Generate Agricultural Reports",
        use_container_width=True,
    )


if submitted:

    if not location.strip() or not intended_crop.strip():
        st.error("Please enter your farm location and intended crop.")

    else:

        farm_data = {
            "location": location,
            "soil_type": soil_type,
            "soil_ph": soil_ph,
            "water_availability": water_availability,
            "season": season,
            "intended_crop": intended_crop,
            "symptoms": symptoms.strip() or "No symptoms provided",
            "growth_stage": growth_stage,
            "weather_information": (
                weather_information.strip() or "Not provided"
            ),
        }

        try:
            with st.spinner(
                "Five AI agents are analyzing your farm. Please wait..."
            ):
                response = requests.post(
                    API_URL,
                    json=farm_data,
                    timeout=180,
                )

            if response.status_code == 200:

                result = response.json()

                st.success("Farm analysis completed!")

                st.header("📊 Your Agricultural Reports")

                reports = [
                    ("🌱 Agent 1: Crop Recommendation",
                     "crop_recommendation"),
                    ("🧪 Agent 2: Soil Health & Nutrients",
                     "soil_health_report"),
                    ("🐛 Agent 3: Pest & Disease",
                     "pest_disease_report"),
                    ("🌦️ Agent 4: Weather & Irrigation",
                     "weather_irrigation_report"),
                    ("📋 Agent 5: Farm Planning & Risk Management",
                     "farm_plan"),
                ]

                for title, key in reports:
                    with st.expander(title, expanded=True):
                        report = result.get(key)
                        st.write(report or "No report available.")

            else:
                st.error(
                    f"Backend error: HTTP {response.status_code}"
                )
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Please make sure the backend server is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The analysis took too long. "
                "Please check the backend terminal."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")


st.divider()

st.caption(
    "Smart Agriculture Advisor | Five-Agent AI System | "
    "AI guidance should be verified with local agricultural experts."
)

