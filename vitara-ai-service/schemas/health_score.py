# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Dict

class NLPResult(BaseModel):
    emotion: str = Field(..., description="Dominant emotion detected (e.g., happy, sad, anxious, neutral)")
    stress_level: float = Field(..., ge=0.0, le=1.0, description="Stress level from journal analysis, scale 0.0 - 1.0")

class FoodResult(BaseModel):
    estimated_calories: int = Field(..., ge=0, description="Estimated calories from food vision model")

class SleepResult(BaseModel):
    quality_score: int = Field(..., ge=0, le=100, description="Sleep quality score from sleep model, scale 0 - 100")

class TypingResult(BaseModel):
    stress_score: float = Field(..., ge=0.0, le=1.0, description="Stress score from keystroke typing dynamics, scale 0.0 - 1.0")

class HealthScoreRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    nlp_result: NLPResult = Field(..., description="Output from NLP journal analysis")
    food_result: FoodResult = Field(..., description="Output from food vision estimation")
    sleep_result: SleepResult = Field(..., description="Output from sleep quality assessment")
    typing_result: TypingResult = Field(..., description="Output from typing stress detection")

class HealthScoreBreakdown(BaseModel):
    mood: int = Field(..., ge=0, le=100, description="Mood sub-score, scale 0 - 100")
    nutrition: int = Field(..., ge=0, le=100, description="Nutrition sub-score, scale 0 - 100")
    stress: int = Field(..., ge=0, le=100, description="Stress sub-score (100 means no stress), scale 0 - 100")
    sleep: int = Field(..., ge=0, le=100, description="Sleep sub-score, scale 0 - 100")

class HealthScoreResponse(BaseModel):
    health_score: int = Field(..., ge=0, le=100, description="Overall consolidated health score, scale 0 - 100")
    breakdown: HealthScoreBreakdown = Field(..., description="Breakdown of sub-scores across different health dimensions")
