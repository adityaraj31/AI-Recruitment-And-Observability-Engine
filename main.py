import os
import asyncio
from dotenv import load_dotenv
from src.graph import app as graph_app
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)

async def main():
    logger.info("Starting AI Recruitment Engine (Graph Mode)...")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in .env file.")
        return
    
    # Load data
    try:
        # Try reading files as bytes correctly
        if os.path.exists("resume.txt"):
             with open("resume.txt", "rb") as f:
                resume_bytes = f.read()
             resume_filename = "resume.txt"
        else:
             logger.warning("resume.txt not found.")
             resume_bytes = None
             resume_filename = None

        if os.path.exists("job_description.txt"):
             with open("job_description.txt", "rb") as f:
                jd_bytes = f.read()
             jd_filename = "job_description.txt"
        else:
             logger.warning("job_description.txt not found.")
             jd_bytes = None
             jd_filename = None
             
    except Exception as e:
        logger.error(f"Error reading files: {e}")
        return

    # Prepare Input
    inputs = {
        "resume_file_bytes": resume_bytes,
        "resume_filename": resume_filename,
        "resume_text": None, # Could pass string directly if needed
        "jd_file_bytes": jd_bytes,
        "jd_filename": jd_filename,
        "jd_text": None,
        "resume_data": {},
        "jd_data": {}
    }

    # Run Graph
    try:
        final_state = await graph_app.ainvoke(inputs)
    except Exception as e:
        logger.error(f"Graph Execution Error: {e}")
        return

    if final_state.get("error"):
        print(f"Error: {final_state['error']}")
        return

    result = final_state.get("analysis", {})
    
    # Output Result
    print("\n" + "="*50)
    print("RECRUITMENT ENGINE RESULTS (LANGGRAPH)")
    print("="*50)
    print(f"Candidate: {result.get('candidate_name', 'Unknown')}")
    print(f"Job Role: {result.get('job_title', 'Unknown')}")
    print(f"Match Score: {result.get('score')}/100")
    print(f"Reasoning: {result.get('reasoning')}")
    print(f"Missing Skills: {', '.join(result.get('missing_skills', []))}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
