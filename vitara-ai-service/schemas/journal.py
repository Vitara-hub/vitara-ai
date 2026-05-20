# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List

class JournalRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Teks jurnal pengguna.")
    user_id: str = Field(None, description="ID unik pengguna untuk menyimpan riwayat kesehatan RAG.")


class JournalResponse(BaseModel):
    emotion: str
    stress_level: float
    topics: List[str]

    