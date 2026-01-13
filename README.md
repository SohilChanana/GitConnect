# GitConnect

A powerful tool that visualizes code dependencies, analyzes impact, and summarizes repositories using a Knowledge Graph and LLMs.

## Features

- **Knowledge Graph**: Stores code structure (Files, Classes, Functions) and relationships (IMPORTS, CALLS, DEFINES) in Neo4j.
- **Vector Search & Recall**: Uses **Moorcheh.ai "Memory-in-a-Box"** for semantic search and efficient file content retrieval.
- **Impact Analysis (GraphRAG)**: detailed answer to "What breaks if I change X?" using hybrid vector + graph traversal.
- **Repository Summarization**: Uses **Google Gemini** detailed viability reports, tech stack analysis, and summaries.
- **Code Parsing**: Uses **Tree-sitter** for accurate parsing of Python and JavaScript codebases.
- **Interactive UI**: React-based frontend for visualizing graphs and interacting with the agent.

## Setup

### 1. Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Neo4j Aura account (Free Tier available)
- API Keys:
  - **OpenAI** (for Embeddings/LLM)
  - **Moorcheh** (for Vector Store)
  - **Google Gemini** (for Summarization)

### 2. Installation

#### Backend

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Frontend

```bash
cd frontend
npm install
```

### 3. Configuration

Copy `.env.template` to `.env` and fill in your credentials:

```bash
cp .env.template .env
```

Edit `.env` with your values:

```ini
# Neo4j
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
AURA_INSTANCEID=...
AURA_INSTANCENAME=...

# AI Providers
OPENAI_API_KEY=sk-...
MOORCHEH_API_KEY=...
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash-exp

# App Settings
ADMIN_SECRET=super-secure-secret-for-dev
GITHUB_TOKEN=... # Optional (for private repos)
```

### 4. Running the Application

**Backend API**:

```bash
# From root directory
uvicorn src.main:app --reload
```
The API will be available at `http://localhost:8000`.

**Frontend**:

```bash
# From frontend directory
npm run dev
```
The UI will be available at `http://localhost:5173`.

## API Endpoints

### Core
- `POST /ingest`: Clone and parse a GitHub repository into Neo4j & Moorcheh.
- `DELETE /repos/{name}`: Remove a repository from the system.
- `GET /health`: Check system status.

### Intelligence
- `POST /analyze`: Ask questions about dependencies (e.g., "What depends on `UserAuth`?").
- `POST /summarize`: Generate a comprehensive summary and viability score for a repository using Gemini.

## Project Structure

```
GitConnect_Test/
├── frontend/              # React/Vite Frontend
│   ├── src/
│   └── package.json
├── src/
│   ├── main.py            # FastAPI entry point
│   ├── config.py          # Settings management
│   ├── graph_manager.py   # Neo4j operations
│   ├── moorcheh_manager.py # Moorcheh vector store operations
│   ├── parser.py          # Tree-sitter parsing logic
│   ├── github_fetcher.py  # Git operations
│   └── summarizer/        # Gemini summarization logic
├── tests/                 # Pytest suite
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## License

MIT
