# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List

class FoodResponse(BaseModel):
    foods: List[str]
    estimated_calories: int
