import os
from dotenv import load_dotenv
from src.agents.resume_parser import parse_resume
from src.agents.jd_parser import parse_jd
from src.agents.ranker import rank_candidate
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)

def main():
    logger.info("Starting AI Recruitment Engine...")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in .env file.")
        return
    
    # Load data (Hardcoded for MVP, or read from files)
    # In a real scenario, these would come from the user or a database
    
    # Sample Data Loading
    try:
        with open("resume.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
        with open("job_description.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
    except FileNotFoundError:
        logger.warning("Sample files not found. Using dummy data.")
        resume_text = "Dummy Resume"
        jd_text = "Dummy JD"

    result = analyze_candidate(resume_text, jd_text)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Output Result
    print("\n" + "="*50)
    print("RECRUITMENT ENGINE RESULTS")
    print("="*50)
    print(f"Candidate: {result.get('candidate_name', 'Unknown')}")
    print(f"Job Role: {result.get('job_title', 'Unknown')}")
    print(f"Match Score: {result.get('score')}/100")
    print(f"Reasoning: {result.get('reasoning')}")
    print(f"Missing Skills: {', '.join(result.get('missing_skills', []))}")
    print("="*50 + "\n")

def analyze_candidate(resume_text: str, jd_text: str) -> dict:
    """
    Orchestrates the resume parsing, JD parsing, and ranking.
    Returns a dictionary with the combined results.
    """
    # 1. Parse Resume
    resume_data = parse_resume(resume_text)
    if "error" in resume_data:
        return {"error": resume_data["error"]}

    # 2. Parse JD
    jd_data = parse_jd(jd_text)
    if "error" in jd_data:
        return {"error": jd_data["error"]}

    # 3. Rank Candidate
    ranking = rank_candidate(resume_data, jd_data)
    if "error" in ranking:
         return {"error": ranking["error"]}
    
    return {
        "candidate_name": resume_data.get("name"),
        "job_title": jd_data.get("job_title"),
        "score": ranking.get("score"),
        "reasoning": ranking.get("reasoning"),
        "missing_skills": ranking.get("missing_skills")
    }

if __name__ == "__main__":
    main()
