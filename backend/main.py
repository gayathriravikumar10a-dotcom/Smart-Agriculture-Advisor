
from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents.agriculture_workflow import build_agriculture_workflow
from rag.rag_pipeline import generate_rag_answer


app = FastAPI(
    title="Smart Agriculture Advisor",
    description="A five-agent AI farming advisory system",
    version="1.0.0",
)


# Farm information received from the frontend
class FarmRequest(BaseModel):
    location: str = Field(min_length=1)
    soil_type: str = Field(min_length=1)
    soil_ph: float = Field(ge=0, le=14)
    water_availability: str = Field(min_length=1)
    season: str = Field(min_length=1)
    intended_crop: str = Field(min_length=1)

    symptoms: str = "No symptoms provided"
    growth_stage: str = "Unknown"
    weather_information: str = "Not provided"


class RAGQuestionRequest(BaseModel):
    question: str = Field(min_length=1)


# Build the five-agent workflow
workflow = build_agriculture_workflow()


@app.get("/")
def home():
    return {
        "message": "Smart Agriculture Advisor API is running!",
        "agents": 5,
        "status": "ready",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "project": "Smart Agriculture Advisor",
    }


@app.post("/analyze-farm")
def analyze_farm(farm: FarmRequest):

    farm_details = farm.model_dump()

    result = workflow.invoke(farm_details)

    return {
        "location": farm.location,
        "intended_crop": farm.intended_crop,

        "crop_recommendation": result.get(
            "crop_recommendation"
        ),

        "soil_health_report": result.get(
            "soil_health_report"
        ),

        "pest_disease_report": result.get(
            "pest_disease_report"
        ),

        "weather_irrigation_report": result.get(
            "weather_irrigation_report"
        ),

        "farm_plan": result.get("farm_plan"),
    }


@app.post("/ask-agriculture")
def ask_agriculture(request: RAGQuestionRequest):
    answer = generate_rag_answer(request.question)
    return {
        "question": request.question,
        "answer": answer,
        "status": "success",
    }
