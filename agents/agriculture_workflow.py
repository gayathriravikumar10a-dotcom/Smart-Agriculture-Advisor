
from langgraph.graph import StateGraph, START, END

from agents.farm_state import FarmState
from agents.crop_recommendation_agent import recommend_crops
from agents.soil_health_agent import analyze_soil
from agents.pest_disease_agent import analyze_pest_disease
from agents.weather_irrigation_agent import analyze_weather_irrigation
from agents.farm_planning_agent import create_farm_plan


def crop_recommendation_node(state: FarmState) -> dict:
    report = recommend_crops(
        soil_type=state["soil_type"],
        soil_ph=state["soil_ph"],
        water_availability=state["water_availability"],
        season=state["season"],
        location=state["location"],
    )
    return {"crop_recommendation": report}


def soil_health_node(state: FarmState) -> dict:
    report = analyze_soil(
        soil_type=state["soil_type"],
        soil_ph=state["soil_ph"],
        crop=state.get("intended_crop", "Not yet selected"),
        location=state["location"],
    )
    return {"soil_health_report": report}


def pest_disease_node(state: FarmState) -> dict:
    report = analyze_pest_disease(
        crop=state.get("intended_crop", "Not yet selected"),
        symptoms=state.get("symptoms", "No symptoms provided"),
        location=state["location"],
        growth_stage=state.get("growth_stage", "Unknown"),
    )
    return {"pest_disease_report": report}


def weather_irrigation_node(state: FarmState) -> dict:
    report = analyze_weather_irrigation(
        crop=state.get("intended_crop", "Not yet selected"),
        location=state["location"],
        soil_type=state["soil_type"],
        water_availability=state["water_availability"],
        growth_stage=state.get("growth_stage", "Unknown"),
        weather_information=state.get(
            "weather_information",
            "Not provided",
        ),
    )
    return {"weather_irrigation_report": report}


def farm_planning_node(state: FarmState) -> dict:
    report = create_farm_plan(
        location=state["location"],
        intended_crop=state.get("intended_crop", "Not yet selected"),
        crop_recommendation=state.get("crop_recommendation", ""),
        soil_health_report=state.get("soil_health_report", ""),
        pest_disease_report=state.get("pest_disease_report", ""),
        weather_irrigation_report=state.get(
            "weather_irrigation_report", ""
        ),
    )
    return {"farm_plan": report}


def build_agriculture_workflow():
    """Build the complete five-agent workflow."""

    graph = StateGraph(FarmState)

    graph.add_node("crop_recommendation", crop_recommendation_node)
    graph.add_node("soil_health", soil_health_node)
    graph.add_node("pest_disease", pest_disease_node)
    graph.add_node("weather_irrigation", weather_irrigation_node)
    graph.add_node("farm_planning", farm_planning_node)

    graph.add_edge(START, "crop_recommendation")
    graph.add_edge("crop_recommendation", "soil_health")
    graph.add_edge("soil_health", "pest_disease")
    graph.add_edge("pest_disease", "weather_irrigation")
    graph.add_edge("weather_irrigation", "farm_planning")
    graph.add_edge("farm_planning", END)

    return graph.compile()


if __name__ == "__main__":
    workflow = build_agriculture_workflow()

    farm_details: FarmState = {
        "location": "Karnataka, India",
        "soil_type": "Loamy",
        "soil_ph": 6.5,
        "water_availability": "Moderate irrigation",
        "season": "Kharif",
        "intended_crop": "Maize",
        "symptoms": "Yellow leaves and small holes on some leaves",
        "growth_stage": "Vegetative stage",
        "weather_information": "No current forecast available",
    }

    result = workflow.invoke(farm_details)

    reports = [
        ("AGENT 1: CROP RECOMMENDATION", "crop_recommendation"),
        ("AGENT 2: SOIL HEALTH AND NUTRIENTS", "soil_health_report"),
        ("AGENT 3: PEST AND PLANT DISEASE", "pest_disease_report"),
        ("AGENT 4: WEATHER AND IRRIGATION", "weather_irrigation_report"),
        ("AGENT 5: FARM PLAN AND RISK MANAGEMENT", "farm_plan"),
    ]

    print("\nSMART AGRICULTURE ADVISOR")
    print("=" * 50)

    for title, key in reports:
        print(f"\n{title}")
        print("-" * 50)
        print(result.get(key, "No report returned."))
