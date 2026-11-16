
from unittest.mock import patch, MagicMock

from agents.crop_recommendation_agent import recommend_crops


def test_recommend_crops_returns_recommendation():
    """Test that the agent returns Gemini's recommendation."""

    mock_response = MagicMock()
    mock_response.text = (
        "Potential crops: rice, maize, and pulses."
    )

    with patch(
        "agents.crop_recommendation_agent.client.models.generate_content",
        return_value=mock_response,
    ) as mock_gemini:

        result = recommend_crops(
            soil_type="Loamy",
            soil_ph=6.5,
            water_availability="Moderate irrigation",
            season="Kharif",
            location="Karnataka, India",
        )

    assert "rice" in result.lower()
    assert "maize" in result.lower()
    mock_gemini.assert_called_once()


def test_recommend_crops_handles_api_error():
    """Test that the agent handles a non-retryable API error."""

    with patch(
        "agents.crop_recommendation_agent.client.models.generate_content",
        side_effect=Exception("Test API error"),
    ):

        result = recommend_crops(
            soil_type="Clay",
            soil_ph=7.0,
            water_availability="Low",
            season="Rabi",
            location="Karnataka, India",
        )

    assert "request failed" in result.lower()
    assert "test api error" in result.lower()

def test_invalid_soil_ph_does_not_call_gemini():
    """Reject invalid pH before making an API request."""

    with patch(
        "agents.crop_recommendation_agent.client.models.generate_content"
    ) as mock_gemini:

        result = recommend_crops(
            soil_type="Loamy",
            soil_ph=15,
            water_availability="Moderate irrigation",
            season="Kharif",
            location="Karnataka, India",
        )

    assert "soil ph must be between 0 and 14" in result.lower()
    mock_gemini.assert_not_called()


def test_empty_soil_type_does_not_call_gemini():
    """Reject an empty soil type before making an API request."""

    with patch(
        "agents.crop_recommendation_agent.client.models.generate_content"
    ) as mock_gemini:

        result = recommend_crops(
            soil_type="   ",
            soil_ph=6.5,
            water_availability="Moderate irrigation",
            season="Kharif",
            location="Karnataka, India",
        )

    assert "soil type cannot be empty" in result.lower()
    mock_gemini.assert_not_called()
