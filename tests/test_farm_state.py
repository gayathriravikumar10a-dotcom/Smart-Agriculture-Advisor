
from agents.farm_state import FarmState


def test_farm_state_accepts_farm_details():
    state: FarmState = {
        "location": "Karnataka, India",
        "soil_type": "Loamy",
        "soil_ph": 6.5,
        "water_availability": "Moderate irrigation",
        "season": "Kharif",
    }

    assert state["location"] == "Karnataka, India"
    assert state["soil_ph"] == 6.5


def test_farm_state_accepts_agent_reports():
    state: FarmState = {
        "crop_recommendation": "Potential crops: maize and pulses.",
        "soil_health_report": "A soil test is recommended.",
        "errors": [],
    }

    assert "maize" in state["crop_recommendation"]
    assert state["errors"] == []
