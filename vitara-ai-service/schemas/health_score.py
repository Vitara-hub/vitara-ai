# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Dict, Optional

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
    nlp_result: Optional[NLPResult] = Field(None, description="Output from NLP journal analysis")
    food_result: Optional[FoodResult] = Field(None, description="Output from food vision estimation")
    sleep_result: Optional[SleepResult] = Field(None, description="Output from sleep quality assessment")
    typing_result: Optional[TypingResult] = Field(None, description="Output from typing stress detection")

class HealthScoreBreakdown(BaseModel):
    mood: Optional[int] = Field(None, ge=0, le=100, description="Mood sub-score, scale 0 - 100")
    nutrition: Optional[int] = Field(None, ge=0, le=100, description="Nutrition sub-score, scale 0 - 100")
    stress: Optional[int] = Field(None, ge=0, le=100, description="Stress sub-score (100 means no stress), scale 0 - 100")
    sleep: Optional[int] = Field(None, ge=0, le=100, description="Sleep sub-score, scale 0 - 100")

class HealthScoreResponse(BaseModel):
    health_score: int = Field(..., ge=0, le=100, description="Overall consolidated health score, scale 0 - 100")
    breakdown: HealthScoreBreakdown = Field(..., description="Breakdown of sub-scores across different health dimensions")
