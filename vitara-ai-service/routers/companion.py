# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
from schemas.companion import CompanionChatRequest
from services.llm_companion import LLMCompanionService

router = APIRouter(prefix="/companion", tags=["Companion"])

# Initialize companion service instance
companion_service = LLMCompanionService()

@router.post("/chat")
async def chat_companion(request: CompanionChatRequest):
    """
    Mengirim pesan percakapan ke asisten AI (Vitara Companion).
    Respons dikirim secara streaming menggunakan format Server-Sent Events (SSE).
    """
    try:
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disables buffering in Nginx/Gunicorn proxies
        }
        return StreamingResponse(
            companion_service.chat_stream(
                user_id=request.user_id,
                user_message=request.message
            ),
            headers=headers,
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan pada layanan LLM Companion: {str(e)}"
        )
