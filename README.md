# GitConnect

A tool that visualizes code dependencies and answers impact-analysis questions using a Knowledge Graph.

## Features

- **Code Parsing**: Uses Tree-sitter to parse Python and JavaScript codebases
- **Knowledge Graph**: Stores code structure in Neo4j with nodes (File, Class, Function) and relationships (IMPORTS, CALLS, DEFINES)
- **GraphRAG**: Answers natural language questions about code dependencies using hybrid vector + graph search

## Setup

### 1. Prerequisites

- Python 3.10+
- Neo4j Aura account (Free Tier available)
- OpenAI API key

### 2. Installation

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

### 3. Configuration

Copy `.env.template` to `.env` and fill in your credentials:

```bash
cp .env.template .env
```

Edit `.env` with your values:
- `NEO4J_URI`: Your Neo4j Aura connection URI
- `NEO4J_USERNAME`: Neo4j username (usually "neo4j")
- `NEO4J_PASSWORD`: Your Neo4j password
- `OPENAI_API_KEY`: Your OpenAI API key
- `GITHUB_TOKEN`: (Optional) GitHub token for private repos

### 4. Running the API

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /ingest
Ingest a GitHub repository into the knowledge graph.

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/repo"}'
```

### POST /analyze
Analyze code dependencies with natural language queries.

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"query": "What breaks if I delete the UserAuth class?"}'
```

### GET /health
Check API and database connection status.

```bash
curl http://localhost:8000/health
```

## Project Structure

```
GitConnect_Test/
├── requirements.txt        # Python dependencies
├── .env.template          # Environment variables template
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   ├── config.py          # Configuration management
│   ├── parser.py          # Tree-sitter AST parsing
│   ├── github_fetcher.py  # GitHub repository fetching
│   ├── graph_manager.py   # Neo4j operations
│   └── models/
│       ├── __init__.py
│       └── entities.py    # Pydantic models
└── tests/
    ├── __init__.py
    ├── test_parser.py
    └── test_graph_manager.py
```

## License

MIT
