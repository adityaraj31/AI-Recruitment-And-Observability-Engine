from typing import TypedDict, Optional, Dict
from langgraph.graph import StateGraph, END
from src.utils.ocr import extract_text_from_file
from src.agents.resume_parser import parse_resume
from src.agents.jd_parser import parse_jd
from src.agents.optimist import get_optimist_opinion
from src.agents.skeptic import get_skeptic_opinion
from src.agents.ranker import rank_candidate
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 1. Define State
class RecruitmentState(TypedDict):
    # Inputs
    resume_file_bytes: Optional[bytes]
    resume_filename: Optional[str]
    resume_text: Optional[str]
    
    jd_file_bytes: Optional[bytes]
    jd_filename: Optional[str]
    jd_text: Optional[str]
    
    # Intermediates
    resume_data: Optional[Dict]
    jd_data: Optional[Dict]
    
    # Opinions (Debate Node)
    optimist_opinion: Optional[str]
    skeptic_opinion: Optional[str]
    
    # Output
    analysis: Optional[Dict]
    error: Optional[str]

# 2. Define Nodes
def ingest_resume(state: RecruitmentState):
    logger.info("Node: Ingest Resume")
    try:
        if state.get("resume_file_bytes"):
            text = extract_text_from_file(state["resume_file_bytes"], state.get("resume_filename", "resume.txt"))
            if text.startswith("Error:"):
                return {"error": text}
            logger.info(f"Ingested resume from file. Length: {len(text)}")
            return {"resume_text": text}
        
        text_input = state.get("resume_text")
        if text_input and text_input.strip() and text_input.lower() != "string":
            logger.info(f"Ingested resume from text input. Length: {len(text_input)}")
            return {"resume_text": text_input}
            
        return {"error": "No valid resume text or file provided"}
    except Exception as e:
        logger.error(f"Ingest Resume Error: {e}")
        return {"error": str(e)}

def ingest_jd(state: RecruitmentState):
    logger.info("Node: Ingest JD")
    try:
        if state.get("jd_file_bytes"):
            text = extract_text_from_file(state["jd_file_bytes"], state.get("jd_filename", "jd.txt"))
            if text.startswith("Error:"):
                return {"error": text}
            logger.info(f"Ingested JD from file. Length: {len(text)}")
            return {"jd_text": text}
        
        text_input = state.get("jd_text")
        if text_input and text_input.strip() and text_input.lower() != "string":
            logger.info(f"Ingested JD from text input. Length: {len(text_input)}")
            return {"jd_text": text_input}
            
        return {"error": "No valid JD text or file provided"}
    except Exception as e:
        logger.error(f"Ingest JD Error: {e}")
        return {"error": str(e)}

def parse_resume_node(state: RecruitmentState):
    logger.info("Node: Parse Resume")
    if state.get("error"): return None
    
    result = parse_resume(state["resume_text"])
    if "error" in result:
        return {"error": result["error"]}
    return {"resume_data": result}

def parse_jd_node(state: RecruitmentState):
    logger.info("Node: Parse JD")
    if state.get("error"): return None
    
    result = parse_jd(state["jd_text"])
    if "error" in result:
        return {"error": result["error"]}
    return {"jd_data": result}

def optimist_node(state: RecruitmentState):
    logger.info("Node: The Optimist")
    if state.get("error"): return None
    
    opinion = get_optimist_opinion(state["resume_data"], state["jd_data"])
    return {"optimist_opinion": opinion}

def skeptic_node(state: RecruitmentState):
    logger.info("Node: The Skeptic")
    if state.get("error"): return None
    
    opinion = get_skeptic_opinion(state["resume_data"], state["jd_data"])
    return {"skeptic_opinion": opinion}

def rank_node(state: RecruitmentState):
    logger.info("Node: Mediator (Final Rank)")
    if state.get("error"): return None
    
    # Wait for both opinions to be present in state
    result = rank_candidate(
        state["resume_data"], 
        state["jd_data"], 
        state.get("optimist_opinion", ""), 
        state.get("skeptic_opinion", "")
    )
    
    if "error" in result:
        return {"error": result["error"]}
    
    result["candidate_name"] = state["resume_data"].get("name", "Unknown")
    result["job_title"] = state["jd_data"].get("job_title", "Unknown")
    
    return {"analysis": result}

# 3. Build Graph
workflow = StateGraph(RecruitmentState)

workflow.add_node("ingest_resume", ingest_resume)
workflow.add_node("ingest_jd", ingest_jd)
workflow.add_node("parse_resume", parse_resume_node)
workflow.add_node("parse_jd", parse_jd_node)
workflow.add_node("optimist", optimist_node)
workflow.add_node("skeptic", skeptic_node)
workflow.add_node("rank", rank_node)

# Define Edges
workflow.set_entry_point("ingest_resume")
workflow.set_entry_point("ingest_jd")

workflow.add_edge("ingest_resume", "parse_resume")
workflow.add_edge("ingest_jd", "parse_jd")

# Once both are parsed, start the debate
workflow.add_edge("parse_resume", "optimist")
workflow.add_edge("parse_resume", "skeptic")
workflow.add_edge("parse_jd", "optimist")
workflow.add_edge("parse_jd", "skeptic")

# Fan-in from debate to mediator
workflow.add_edge("optimist", "rank")
workflow.add_edge("skeptic", "rank")

# Compile
app = workflow.compile()
