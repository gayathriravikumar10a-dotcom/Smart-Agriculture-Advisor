
from unittest.mock import MagicMock, patch

from agents.weather_irrigation_agent import analyze_weather_irrigation


def test_valid_input_returns_report():
    mock_response = MagicMock()
    mock_response.text = "Check rainfall before deciding to irrigate."

    with patch(
        "agents.weather_irrigation_agent.client.models.generate_content",
        return_value=mock_response,
    ) as mock_gemini:
        result = analyze_weather_irrigation(
            crop="Maize",
            location="Karnataka, India",
            soil_type="Loamy",
            water_availability="Moderate irrigation",
            growth_stage="Vegetative stage",
        )

    assert "Check rainfall" in result
    mock_gemini.assert_called_once()


def test_empty_crop_does_not_call_gemini():
    with patch(
        "agents.weather_irrigation_agent.client.models.generate_content"
    ) as mock_gemini:
        result = analyze_weather_irrigation(
            crop="",
            location="Karnataka, India",
            soil_type="Loamy",
            water_availability="Moderate irrigation",
        )

    assert "crop cannot be empty" in result.lower()
    mock_gemini.assert_not_called()


def test_empty_location_does_not_call_gemini():
    with patch(
        "agents.weather_irrigation_agent.client.models.generate_content"
    ) as mock_gemini:
        result = analyze_weather_irrigation(
            crop="Maize",
            location="",
            soil_type="Loamy",
            water_availability="Moderate irrigation",
        )

    assert "location cannot be empty" in result.lower()
    mock_gemini.assert_not_called()


def test_quota_error_is_handled():
    with patch(
        "agents.weather_irrigation_agent.client.models.generate_content",
        side_effect=Exception("RESOURCE_EXHAUSTED: quota exceeded"),
    ):
        result = analyze_weather_irrigation(
            crop="Maize",
            location="Karnataka, India",
            soil_type="Loamy",
            water_availability="Moderate irrigation",
        )

    assert "quota is currently exhausted" in result.lower()
