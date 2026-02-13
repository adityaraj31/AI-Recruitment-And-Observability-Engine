from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

@traceable(name="optimist_agent")
def get_optimist_opinion(resume_data: dict, jd_data: dict) -> str:
    """
    Analyzes the candidate from an optimistic perspective, focusing on 
    potential, transferrable skills, and growth.
    """
    logger.info("Optimist Agent: Analyzing candidate potential...")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are 'The Optimist', a talent scout who sees the best in every candidate. "
            "Your goal is to advocate for why this candidate SHOULD be hired. "
            "Look for transferrable skills, passion, educational background, and signs of growth. "
            "Even if they lack a specific skill, explain how their existing experience suggests they can learn it quickly."
        )),
        ("human", (
            "Resume Details: {resume_data}\n\n"
            "Job Requirements: {jd_data}\n\n"
            "Provide your optimistic evaluation of why this candidate is a great fit."
        ))
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "resume_data": str(resume_data),
            "jd_data": str(jd_data)
        })
        return response.content
    except Exception as e:
        logger.error(f"Optimist Agent Error: {e}")
        return f"Error in optimistic evaluation: {str(e)}"
