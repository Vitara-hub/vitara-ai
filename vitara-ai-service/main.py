# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
import uvicorn
from routers import journal, food

app = FastAPI(
    title="Vitara AI Service",
    description="API Service for Vitara AI Models (NLP, Vision)",
    version="1.0.0"
)

app.include_router(journal.router)
app.include_router(food.router)

@app.get("/")
async def root():
    return {"message": "Vitara AI Service is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
