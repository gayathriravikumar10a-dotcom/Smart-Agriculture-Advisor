from unittest.mock import patch

from agents.agriculture_workflow import build_agriculture_workflow


def test_five_agent_workflow_runs_without_gemini():
    """Verify that all five agents run using mocked responses."""

    with (
        patch(
            "agents.agriculture_workflow.recommend_crops",
            return_value="Mock crop recommendation",
        ) as mock_crop,
        patch(
            "agents.agriculture_workflow.analyze_soil",
            return_value="Mock soil health report",
        ) as mock_soil,
        patch(
            "agents.agriculture_workflow.analyze_pest_disease",
            return_value="Mock pest disease report",
        ) as mock_pest,
        patch(
            "agents.agriculture_workflow.analyze_weather_irrigation",
            return_value="Mock weather irrigation report",
        ) as mock_weather,
        patch(
            "agents.agriculture_workflow.create_farm_plan",
            return_value="Mock final farm plan",
        ) as mock_plan,
    ):
        workflow = build_agriculture_workflow()

        result = workflow.invoke({
            "location": "Karnataka, India",
            "soil_type": "Loamy",
            "soil_ph": 6.5,
            "water_availability": "Moderate irrigation",
            "season": "Kharif",
            "intended_crop": "Maize",
            "symptoms": "Yellow leaves",
            "growth_stage": "Vegetative stage",
            "weather_information": "No current forecast available",
        })

    assert result["crop_recommendation"] == "Mock crop recommendation"
    assert result["soil_health_report"] == "Mock soil health report"
    assert result["pest_disease_report"] == "Mock pest disease report"
    assert result["weather_irrigation_report"] == (
        "Mock weather irrigation report"
    )
    assert result["farm_plan"] == "Mock final farm plan"

    mock_crop.assert_called_once()
    mock_soil.assert_called_once()
    mock_pest.assert_called_once()
    mock_weather.assert_called_once()
    mock_plan.assert_called_once()
