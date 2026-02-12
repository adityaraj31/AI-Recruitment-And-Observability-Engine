from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from src.utils.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

# Define the structured output for JD
class JobDescriptionData(BaseModel):
    job_title: str = Field(description="Title of the job")
    required_skills: List[str] = Field(description="List of mandatory technical and soft skills")
    min_experience_years: float = Field(description="Minimum years of experience required")
    preferred_qualifications: List[str] = Field(description="List of preferred qualifications or 'nice-to-haves'")

@traceable(name="parse_jd")
def parse_jd(jd_text: str) -> dict:
    """
    Parses a job description text and extracts structured data.
    """
    logger.info("Parsing job description...")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=JobDescriptionData)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR assistant. Extract structured requirements from the following Job Description.\n{format_instructions}"),
        ("human", "{jd_text}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "jd_text": jd_text,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Job description parsed successfully.")
        return result
    except Exception as e:
        logger.error(f"Error parsing JD: {e}")
        return {"error": str(e)}
