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
        resume_text = """
        John Doe
        Software Engineer
        Skills: Python, Docker, Kubernetes, LangChain
        Experience: 4 years working on backend systems and AI agents.
        """
        jd_text = """
        Job Title: Senior Python AI Engineer
        Requirements:
        - 3+ years of Python experience
        - Experience with LLMs and LangChain
        - Knowledge of Docker and Kubernetes
        """

    # 1. Parse Resume
    resume_data = parse_resume(resume_text)
    if "error" in resume_data:
        return

    # 2. Parse JD
    jd_data = parse_jd(jd_text)
    if "error" in jd_data:
        return

    # 3. Rank Candidate
    ranking = rank_candidate(resume_data, jd_data)
    
    # Output Result
    print("\n" + "="*50)
    print("RECRUITMENT ENGINE RESULTS")
    print("="*50)
    print(f"Candidate: {resume_data.get('name', 'Unknown')}")
    print(f"Job Role: {jd_data.get('job_title', 'Unknown')}")
    print(f"Match Score: {ranking.get('score')}/100")
    print(f"Reasoning: {ranking.get('reasoning')}")
    print(f"Missing Skills: {', '.join(ranking.get('missing_skills', []))}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
