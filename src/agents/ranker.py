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

@traceable(name="rank_candidate")
def rank_candidate(resume_data: dict, jd_data: dict) -> dict:
    """
    Compares extracted Resume data against Job Description data to rank the candidate.
    """
    logger.info("Ranking candidate...")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=RankingOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Technical Recruiter. Evaluate the candidate's Resume against the Job Description. Be strict but fair.\n{format_instructions}"),
        ("human", "RESUME DATA:\n{resume_data}\n\nJOB DESCRIPTION DATA:\n{jd_data}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "resume_data": str(resume_data),
            "jd_data": str(jd_data),
            "format_instructions": parser.get_format_instructions()
        })
        logger.info(f"Candidate ranked: Score {result.get('score')}")
        return result
    except Exception as e:
        logger.error(f"Error ranking candidate: {e}")
        return {"error": str(e)}
