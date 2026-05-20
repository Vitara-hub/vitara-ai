# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class TypingPredictRequest(BaseModel):
    wpm: float = Field(
        ..., 
        ge=0.0, 
        description="Kecepatan mengetik dalam kata per menit"
    )
    backspace_rate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Rasio tombol backspace terhadap total penekanan tombol (0.0 – 1.0)"
    )
    inter_key_timings: List[float] = Field(
        ..., 
        min_length=1,
        description="Array interval waktu antar tombol dalam milidetik"
    )
    user_id: Optional[str] = Field(
        None, 
        description="ID unik pengguna untuk menyimpan riwayat kesehatan RAG."
    )

    @field_validator("inter_key_timings")
    @classmethod
    def check_timings(cls, v: List[float]) -> List[float]:
        if any(t < 0.0 for t in v):
            raise ValueError("Setiap interval waktu antar tombol (inter_key_timings) harus bernilai non-negatif (>= 0.0)")
        return v


class TypingPredictResponse(BaseModel):
    stress_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Tingkat stres dari pola pengetikan, skala 0.0 – 1.0"
    )
