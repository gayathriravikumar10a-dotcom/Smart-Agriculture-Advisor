
from unittest.mock import patch

from agents.soil_health_agent import analyze_soil


def test_valid_soil_input_returns_report():
    """Check that a valid input returns a soil report."""

    with patch(
        "agents.soil_health_agent.client.models.generate_content"
    ) as mock_gemini:

        mock_gemini.return_value.text = (
            "A soil test can help identify nutrient deficiencies."
        )

        result = analyze_soil(
            soil_type="Loamy",
            soil_ph=6.5,
            crop="Maize",
            location="Karnataka, India",
        )

    assert "soil test" in result.lower()
    mock_gemini.assert_called_once()


def test_invalid_soil_ph_does_not_call_gemini():
    """Check that invalid pH is rejected."""

    with patch(
        "agents.soil_health_agent.client.models.generate_content"
    ) as mock_gemini:

        result = analyze_soil(
            soil_type="Loamy",
            soil_ph=15,
            crop="Maize",
            location="Karnataka, India",
        )

    assert "soil ph must be between 0 and 14" in result.lower()
    mock_gemini.assert_not_called()


def test_empty_crop_does_not_call_gemini():
    """Check that an empty crop is rejected."""

    with patch(
        "agents.soil_health_agent.client.models.generate_content"
    ) as mock_gemini:

        result = analyze_soil(
            soil_type="Loamy",
            soil_ph=6.5,
            crop=" ",
            location="Karnataka, India",
        )

    assert "crop cannot be empty" in result.lower()
    mock_gemini.assert_not_called()
