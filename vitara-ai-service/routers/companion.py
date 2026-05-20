# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.companion import CompanionChatRequest, CompanionChatResponse
from services.llm_companion import LLMCompanionService

router = APIRouter(prefix="/companion", tags=["Companion"])

# Initialize companion service instance
companion_service = LLMCompanionService()

@router.post("/chat", response_model=CompanionChatResponse)
async def chat_companion(request: CompanionChatRequest):
    """
    Mengirim pesan percakapan ke asisten AI (Vitara Companion).
    Sistem akan mencari riwayat memori relevan dari ChromaDB (RAG) untuk merespons secara personal.
    """
    try:
        response_data = await companion_service.chat(
            user_id=request.user_id,
            user_message=request.message
        )
        return CompanionChatResponse(
            response=response_data["response"],
            recommendations=response_data["recommendations"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan pada layanan LLM Companion: {str(e)}"
        )
