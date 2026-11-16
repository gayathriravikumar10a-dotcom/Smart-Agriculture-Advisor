
from unittest.mock import patch, MagicMock

from agents.pest_disease_agent import analyze_pest_disease


def test_valid_input_returns_report():
    """Agent returns a report when Gemini responds."""

    mock_response = MagicMock()
    mock_response.text = (
        "Possible causes include pests or disease. "
        "Monitor the crop and consult an agricultural expert."
    )

    with patch(
        "agents.pest_disease_agent.client.models.generate_content",
        return_value=mock_response,
    ) as mock_gemini:

        result = analyze_pest_disease(
            crop="Maize",
            symptoms="Yellow leaves and small holes",
            location="Karnataka, India",
            growth_stage="Vegetative stage",
        )

    assert "Possible causes" in result
    mock_gemini.assert_called_once()


def test_empty_crop_does_not_call_gemini():
    """An empty crop name should be rejected."""

    with patch(
        "agents.pest_disease_agent.client.models.generate_content"
    ) as mock_gemini:

        result = analyze_pest_disease(
            crop="",
            symptoms="Yellow leaves",
            location="Karnataka, India",
        )

    assert "crop cannot be empty" in result.lower()
    mock_gemini.assert_not_called()


def test_empty_symptoms_does_not_call_gemini():
    """Empty symptoms should be rejected."""

    with patch(
        "agents.pest_disease_agent.client.models.generate_content"
    ) as mock_gemini:

        result = analyze_pest_disease(
            crop="Maize",
            symptoms="",
            location="Karnataka, India",
        )

    assert "symptoms cannot be empty" in result.lower()
    mock_gemini.assert_not_called()


def test_gemini_quota_error_is_handled():
    """A quota error should not crash the agent."""

    with patch(
        "agents.pest_disease_agent.client.models.generate_content",
        side_effect=Exception("RESOURCE_EXHAUSTED: quota exceeded"),
    ):

        result = analyze_pest_disease(
            crop="Maize",
            symptoms="Yellow leaves",
            location="Karnataka, India",
        )

    assert "analysis failed" in result.lower()
