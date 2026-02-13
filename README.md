# AI Recruitment & Observability Engine

An automated talent acquisition pipeline designed to streamline candidate screening using Agentic AI. This project uses **LangGraph** to orchestrate a multi-agent workflow that parses resumes/JDs and ranks candidates with full observability via **LangSmith**.

## Features

- **Multi-Agent Orchestration**: Managed by LangGraph for parallel ingestion and state-aware processing.
- **The "Debate Node" Pattern**: A unique consensus-driven evaluation where an **Optimist agent** and a **Skeptic agent** debate a candidate's fit before a **Mediator** reaches a final score.
- **Robust OCR**: Supports parsing text from **PDFs** (digital and scanned), **Images (JPG/PNG)**, and **Text** files.
- **Resume Parsing**: Extracts structured data (skills, experience, education) using Vision-capable LLMs.
- **Job Description Analysis**: Automatically extracts key requirements and qualifications from job posts.
- **Observability**: Full tracing and monitoring of every agent decision in LangSmith.
- **REST API**: FastAPI-based interface with support for file uploads and direct text input.

## Tech Stack

- **Orchestration**: LangGraph
- **LLM Framework**: LangChain
- **Models**: OpenAI (GPT-4o) / OpenRouter (Llama 3.2 Vision, Gemini 2.0 Flash)
- **API**: FastAPI & Uvicorn
- **OCR/PDF**: pypdf, pdf2image, pillow
- **Observability**: LangSmith
- **Package Manager**: uv

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adityarajsingh31/ai-recruitment-engine.git
   cd ai-recruitment-engine
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   # OR
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file from `.env.example`:
   ```bash
   OPENAI_API_KEY=your_key
   LANGCHAIN_API_KEY=your_key
   ```

## Usage

### API Mode (Recommended)
Start the server:
```bash
uvicorn src.api:app --reload --port 8000
```
Access the interactive docs at `http://localhost:8000/docs`.

### CLI Mode
```bash
python main.py
```

## API Documentation

### Analyze Candidate (`POST /analyze`)
This endpoint orchestrates the full graph: Ingestion -> Parsing -> Debate -> Ranking.

#### 1. Uploading Files (Recommended)
Supports PDF and Images for resumes.
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "resume_file=@/path/to/resume.pdf" \
     -F "jd_text=Looking for a Python Engineer with LangGraph experience."
```

#### 2. Direct Text Input
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "resume_text=John Doe. Python expert." \
     -F "jd_text=Senior Python Developer."
```

#### Response Example
```json
{
  "score": 85,
  "reasoning": "The candidate has strong Python experience as noted by the Optimist, though the Skeptic pointed out a lack of Docker...",
  "missing_skills": ["Docker", "Kubernetes"],
  "candidate_name": "John Doe",
  "job_title": "Senior Python Developer"
}
```

## Project Structure

```
ai-recruitment-engine/
├── src/
│   ├── agents/          # Specialized agents (Optimist, Skeptic, Mediator)
│   ├── utils/           # Utilities (OCR, Logger)
│   ├── graph.py         # LangGraph workflow definition
│   └── api.py           # FastAPI endpoints
├── tests/               # API and unit tests
├── requirements.txt
└── README.md
```
