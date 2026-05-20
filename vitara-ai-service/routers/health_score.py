# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException
from schemas.health_score import HealthScoreRequest, HealthScoreResponse
from services.health_score_service import HealthScoreService
from services.llm_companion import memory_store

router = APIRouter(prefix="/health", tags=["Health Score"])

@router.post("/score", response_model=HealthScoreResponse)
async def get_health_score(request: HealthScoreRequest):
    """
    Menghitung skor kesehatan keseluruhan secara rule-based (deterministic)
    berdasarkan sub-hasil analisis mood (NLP), nutrisi (makanan), tidur, dan stres pengetikan.
    """
    try:
        response = HealthScoreService.calculate_health_score(request)
        
        # Simpan riwayat perhitungan skor kesehatan ke memori RAG ChromaDB
        if request.user_id:
            memory_store.add_memory(
                user_id=request.user_id,
                text=(
                    f"Perhitungan Skor Kesehatan Keseluruhan: {response.health_score}/100. "
                    f"Rincian dimensi kesehatan: Mood={response.breakdown.mood}/100, "
                    f"Nutrisi={response.breakdown.nutrition}/100, "
                    f"Bebas Stres={response.breakdown.stress}/100, "
                    f"Tidur={response.breakdown.sleep}/100."
                ),
                mem_type="health_score"
            )
            
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menghitung health score: {str(e)}"
        )

