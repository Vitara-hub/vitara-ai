# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.health_score import HealthScoreRequest, HealthScoreResponse
from services.health_score_service import HealthScoreService

router = APIRouter(prefix="/health", tags=["Health Score"])

@router.post("/score", response_model=HealthScoreResponse)
async def get_health_score(request: HealthScoreRequest):
    """
    Menghitung skor kesehatan keseluruhan secara rule-based (deterministic)
    berdasarkan sub-hasil analisis mood (NLP), nutrisi (makanan), tidur, dan stres pengetikan.
    """
    try:
        response = HealthScoreService.calculate_health_score(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menghitung health score: {str(e)}"
        )
