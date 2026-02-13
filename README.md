# AI Recruitment & Observability Engine

An automated talent acquisition pipeline designed to streamline candidate screening using Agentic AI. This project uses LLMs to parse resumes and job descriptions, ranking candidates based on semantic alignment, with full observability provided by LangSmith.

## Features

- **Resume Parsing**: Extracts structured data (skills, experience, education) from resumes.
- **Job Description Analysis**: Extracts key requirements and qualifications from JDs.
- **Candidate Ranking**: Semantically matches candidates to jobs and provides a compatibility score (0-100) with reasoning.
- **REST API**: FastAPI-based interface for integration.
- **Observability**: Integrated with LangSmith for tracing, debugging, and monitoring LLM interactions.

## Tech Stack

- **Language**: Python
- **LLM Orchestration**: LangChain & LangGraph
- **API Framework**: FastAPI
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
   Create a `.env` file in the root directory.

## Usage

### CLI Mode
```bash
python main.py
```

### API Mode
Start the server:
```bash
uvicorn src.api:app --reload
```
Access docs at `http://localhost:8000/docs`.

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
│   ├── api.py
│   └── main.py
├── tests/
├── requirements.txt
├── .env
└── README.md
```
