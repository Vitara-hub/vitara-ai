# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional

class SleepPredictRequest(BaseModel):
    duration_hours: float = Field(
        ..., 
        ge=0.0, 
        le=24.0, 
        description="Total durasi tidur dalam jam"
    )
    bedtime: str = Field(
        ..., 
        pattern=r"^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$", 
        description="Waktu mulai tidur, format HH:MM (00:00 - 23:59)"
    )
    wake_time: str = Field(
        ..., 
        pattern=r"^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$", 
        description="Waktu bangun, format HH:MM (00:00 - 23:59)"
    )
    interruptions: int = Field(
        ..., 
        ge=0, 
        description="Jumlah kali terbangun di malam hari"
    )
    sleep_debt_hours: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=24.0, 
        description="Akumulasi utang tidur dalam jam"
    )
    user_id: Optional[str] = Field(
        None, 
        description="ID unik pengguna untuk menyimpan riwayat kesehatan RAG."
    )


class SleepPredictResponse(BaseModel):
    quality_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Skor kualitas tidur, skala 0.0 – 1.0"
    )
