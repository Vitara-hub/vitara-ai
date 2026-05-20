# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List

class CompanionChatRequest(BaseModel):
    user_id: str = Field(..., description="ID unik pengguna untuk membedakan riwayat memori RAG.")
    message: str = Field(..., min_length=1, description="Pesan percakapan yang dikirim oleh pengguna.")

class CompanionChatResponse(BaseModel):
    response: str = Field(..., description="Tanggapan percakapan utama dari asisten AI.")
    recommendations: List[str] = Field(..., description="Daftar saran tindakan kesehatan konkret berbasis personalisasi.")
