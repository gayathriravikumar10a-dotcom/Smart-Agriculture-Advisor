
from unittest.mock import MagicMock, patch

from agents.farm_planning_agent import create_farm_plan


def test_valid_reports_return_farm_plan():
    mock_response = MagicMock()
    mock_response.text = "Farm plan: test soil and monitor crops."

    with patch(
        "agents.farm_planning_agent.client.models.generate_content",
        return_value=mock_response,
    ) as mock_gemini:
        result = create_farm_plan(
            location="Karnataka, India",
            intended_crop="Maize",
            crop_recommendation="Potential crop: maize.",
            soil_health_report="A soil test is recommended.",
            pest_disease_report="Monitor leaves for symptoms.",
            weather_irrigation_report="Check rainfall before irrigation.",
        )

    assert "Farm plan" in result
    mock_gemini.assert_called_once()


def test_empty_location_does_not_call_gemini():
    with patch(
        "agents.farm_planning_agent.client.models.generate_content"
    ) as mock_gemini:
        result = create_farm_plan(
            location="",
            intended_crop="Maize",
            crop_recommendation="Crop report",
            soil_health_report="Soil report",
            pest_disease_report="Pest report",
            weather_irrigation_report="Weather report",
        )

    assert "location cannot be empty" in result.lower()
    mock_gemini.assert_not_called()


def test_missing_report_does_not_call_gemini():
    with patch(
        "agents.farm_planning_agent.client.models.generate_content"
    ) as mock_gemini:
        result = create_farm_plan(
            location="Karnataka, India",
            intended_crop="Maize",
            crop_recommendation="Crop report",
            soil_health_report="",
            pest_disease_report="Pest report",
            weather_irrigation_report="Weather report",
        )

    assert "soil health report is missing" in result.lower()
    mock_gemini.assert_not_called()


def test_quota_error_is_handled():
    with patch(
        "agents.farm_planning_agent.client.models.generate_content",
        side_effect=Exception("RESOURCE_EXHAUSTED: quota exceeded"),
    ):
        result = create_farm_plan(
            location="Karnataka, India",
            intended_crop="Maize",
            crop_recommendation="Crop report",
            soil_health_report="Soil report",
            pest_disease_report="Pest report",
            weather_irrigation_report="Weather report",
        )

    assert "quota is currently exhausted" in result.lower()
