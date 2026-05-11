# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List

class JournalRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Teks jurnal pengguna.")

class JournalResponse(BaseModel):
    emotion: str
    stress_level: float
    topics: List[str]

    