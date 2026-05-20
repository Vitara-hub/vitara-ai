# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
import uvicorn
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

from routers import journal, food, health_score, companion, sleep, typing


APP_ENV = os.getenv("APP_ENV", "development")
APP_PORT = int(os.getenv("APP_PORT", 8000))

# Validasi API Key Gemini untuk LLM Companion
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print("\n  [WARNING] GEMINI_API_KEY is not set or is still the placeholder value in .env!")
    print("  Please configure GEMINI_API_KEY in 'vitara-ai-service/.env' to enable the LLM Companion feature.\n")
else:
    print("\n  [INFO] GEMINI_API_KEY loaded successfully.\n")

app = FastAPI(

    title="Vitara AI Service",
    description="API Service for Vitara AI Models (NLP, Vision, Typing, Sleep, LLM)",
    version="1.0.0",
    docs_url="/docs" if APP_ENV == "development" else None,
    redoc_url="/redoc" if APP_ENV == "development" else None
)

app.include_router(journal.router)
app.include_router(food.router)
app.include_router(health_score.router)
app.include_router(companion.router)
app.include_router(sleep.router)
app.include_router(typing.router)


@app.get("/")
async def root():
    return {
        "message": "Vitara AI Service is running",
        "environment": APP_ENV
    }

if __name__ == "__main__":
    # Menjalankan server menggunakan port dari .env
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=APP_PORT, 
        reload=True if APP_ENV == "development" else False
    )
