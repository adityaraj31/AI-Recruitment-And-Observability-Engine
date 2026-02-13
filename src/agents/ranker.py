from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from src.utils.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

class RankingOutput(BaseModel):
    score: int = Field(description="Match score between 0 and 100")
    reasoning: str = Field(description="Detailed reasoning for the score, comparing skills and experience")
    missing_skills: list[str] = Field(description="List of required skills missing from the candidate's profile")

@traceable(name="mediator_rank_candidate")
def rank_candidate(resume_data: dict, jd_data: dict, optimist_opinion: str = "", skeptic_opinion: str = "") -> dict:
    """
    Acts as a Mediator, analyzing the candidate by considering both 
    Optimistic and Skeptical perspectives to reach a fair final score.
    """
    logger.info("Mediator Agent: Analyzing debate to reach consensus...")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=RankingOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are 'The Mediator', an expert Hiring Manager. Your goal is to review a candidate's resume "
            "against a job description, while CONSIDERING the evaluations from two of your specialized agents: "
            "the Optimist (who highlights potential) and the Skeptic (who identifies risk).\n"
            "Your task is to reach a final, balanced decision based on all evidence.\n{format_instructions}"
        )),
        ("human", (
            "RESUME DATA:\n{resume_data}\n\n"
            "JOB DESCRIPTION DATA:\n{jd_data}\n\n"
            "--- AGENT DEBATE ---\n"
            "OPTIMIST'S VIEW: {optimist_opinion}\n\n"
            "SKEPTIC'S VIEW: {skeptic_opinion}\n\n"
            "Provide the final matching analysis."
        ))
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "resume_data": str(resume_data),
            "jd_data": str(jd_data),
            "optimist_opinion": optimist_opinion,
            "skeptic_opinion": skeptic_opinion,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info(f"Final Decision reached: Score {result.get('score')}")
        return result
    except Exception as e:
        logger.error(f"Mediator Error: {e}")
        return {"error": str(e)}
