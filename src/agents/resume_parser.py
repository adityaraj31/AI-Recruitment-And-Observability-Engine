from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from src.utils.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

# Define the structured output for Resume
class ResumeData(BaseModel):
    name: Optional[str] = Field(description="Name of the candidate")
    email: Optional[str] = Field(description="Email address")
    skills: List[str] = Field(description="List of technical and soft skills")
    experience_years: float = Field(description="Total years of professional experience")
    education: List[str] = Field(description="List of degrees and universities")
    recent_role: Optional[str] = Field(description="Most recent job title")

@traceable(name="parse_resume")
def parse_resume(resume_text: str) -> dict:
    """
    Parses a resume text and extracts structured data.
    """
    logger.info("Parsing resume...")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=ResumeData)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR assistant. Extract structured information from the following resume text.\n{format_instructions}"),
        ("human", "{resume_text}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "resume_text": resume_text,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Resume parsed successfully.")
        return result
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        return {"error": str(e)}
