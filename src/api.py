from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import analyze_candidate
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(title="AI Recruitment Engine", version="1.0.0")

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description_text: str

class AnalysisResponse(BaseModel):
    candidate_name: str | None
    job_title: str | None
    score: int | None
    reasoning: str | None
    missing_skills: list[str] | None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    logger.info("Received analysis request")
    try:
        result = analyze_candidate(request.resume_text, request.job_description_text)
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
