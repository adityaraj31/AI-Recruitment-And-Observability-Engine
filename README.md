# AI Recruitment & Observability Engine

An automated talent acquisition pipeline designed to streamline candidate screening using Agentic AI. This project uses LLMs to parse resumes and job descriptions, ranking candidates based on semantic alignment, with full observability provided by LangSmith.

## Features

- **Resume Parsing**: Extracts structured data (skills, experience, education) from resumes.
- **Job Description Analysis**: Extracts key requirements and qualifications from JDs.
- **Candidate Ranking**: semantically matches candidates to jobs and provides a compatibility score (0-100) with reasoning.
- **Observability**: Integrated with LangSmith for tracing, debugging, and monitoring LLM interactions.

## Tech Stack

- **Language**: Python
- **LLM Orchestration**: LangChain
- **LLM Provider**: OpenAI / OpenRouter
- **Observability**: LangSmith
- **Package Management**: uv

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adityarajsingh31/ai-recruitment-engine.git
   cd ai_recruitment_engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # OR if using uv
   uv sync
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory (see `.env.example`):
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_API_BASE=https://openrouter.ai/api/v1  # If using OpenRouter
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY=lsv2-...
   LANGCHAIN_PROJECT="ai-recruitment-engine"
   ```

## Usage

1. **Prepare Data**:
   Place your resume text in `resume.txt` and job description in `job_description.txt` in the root directory.

2. **Run the Engine**:
   ```bash
   python -m src.main
   ```

3. **View Results**:
   The application will output the extracted data and the ranking score to the console.
   Check your LangSmith dashboard to see the full trace of the reasoning process.

## Project Structure

```
ai-recruitment-engine/
├── src/
│   ├── agents/
│   │   ├── resume_parser.py
│   │   ├── jd_parser.py
│   │   └── ranker.py
│   ├── utils/
│   │   └── logger.py
│   └── main.py
├── requirements.txt
├── .env
└── README.md
```
