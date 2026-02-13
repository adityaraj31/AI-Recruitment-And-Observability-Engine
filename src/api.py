from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from src.graph import app as graph_app
from src.utils.logger import setup_logger

load_dotenv()

logger = setup_logger(__name__)

app = FastAPI(title="AI Recruitment Engine", version="2.0.0")

class AnalysisResponse(BaseModel):
    candidate_name: str | None = None
    job_title: str | None = None
    score: int | None = None
    reasoning: str | None = None
    missing_skills: list[str] | None = None
    error: str | None = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    resume_file: Optional[UploadFile] = File(None),
    jd_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None)
):
    logger.info("Received analysis request via Graph")
    
    # Read file content if provided
    resume_bytes = await resume_file.read() if resume_file else None
    jd_bytes = await jd_file.read() if jd_file else None
    
    resume_filename = resume_file.filename if resume_file else None
    jd_filename = jd_file.filename if jd_file else None

    # Prepare State Input
    inputs = {
        "resume_file_bytes": resume_bytes,
        "resume_filename": resume_filename,
        "resume_text": resume_text,
        "jd_file_bytes": jd_bytes,
        "jd_filename": jd_filename,
        "jd_text": jd_text,
        "resume_data": {},
        "jd_data": {}
    }

    try:
        # Invoke Graph
        final_state = await graph_app.ainvoke(inputs)
        
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])
            
        analysis = final_state.get("analysis", {})
        if not analysis:
             raise HTTPException(status_code=500, detail="Analysis failed to produce results.")
             
        return analysis
        
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
