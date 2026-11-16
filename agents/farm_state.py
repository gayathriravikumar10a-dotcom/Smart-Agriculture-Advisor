
from typing import TypedDict


class FarmState(TypedDict, total=False):
    """Shared information passed between agriculture agents."""

    # Farm details
    location: str
    soil_type: str
    soil_ph: float
    water_availability: str
    season: str
    intended_crop: str

    # Pest and disease observations
    symptoms: str
    growth_stage: str

    # Weather information supplied by the user
    weather_information: str

    # Agent reports
    crop_recommendation: str
    soil_health_report: str
    pest_disease_report: str
    weather_irrigation_report: str
    farm_plan: str

    # Workflow errors
    errors: list[str]
