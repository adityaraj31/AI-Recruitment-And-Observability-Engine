from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

@traceable(name="skeptic_agent")
def get_skeptic_opinion(resume_data: dict, jd_data: dict) -> str:
    """
    Analyzes the candidate from a skeptical perspective, focusing on 
    skill gaps, risks, and missing requirements.
    """
    logger.info("Skeptic Agent: Identifying risks and gaps...")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are 'The Skeptic', a rigorous recruiter who focuses on hard requirements and risk. "
            "Your goal is to identify why this candidate might NOT be a perfect fit. "
            "Focus on missing technical skills, tenure gaps, lack of specific industry experience, "
            "or requirements in the JD that are not explicitly proven in the resume. "
            "Be critical but professional."
        )),
        ("human", (
            "Resume Details: {resume_data}\n\n"
            "Job Requirements: {jd_data}\n\n"
            "Provide your skeptical evaluation of the risks and missing skills for this candidate."
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
        logger.error(f"Skeptic Agent Error: {e}")
        return f"Error in skeptical evaluation: {str(e)}"
