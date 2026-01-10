"""FastAPI application for GitConnect.

Provides REST API endpoints for:
- Repository ingestion (/ingest)
- Impact analysis (/analyze)
- Health checks (/health)

Uses hybrid GraphRAG approach:
1. Vector search for semantic similarity
2. Graph traversal for structural dependencies
3. LLM summarization for impact analysis
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from pydantic import BaseModel, Field, validator
import pydantic

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate

from src.config import get_settings
from src.github_fetcher import GitHubFetcher, GitHubFetchError, fetch_repository
from src.parser import CodeParser, parse_repository
from src.graph_manager import GraphManager, GraphConnectionError

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting GitConnect API...")
    settings = get_settings()
    
    # Initialize graph manager
    app.state.graph_manager = GraphManager()
    try:
        app.state.graph_manager.connect()
        app.state.graph_manager.setup_schema()
    except GraphConnectionError as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        # Continue anyway - will fail on first request
    
    # Initialize LangChain components
    app.state.embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
    )
    app.state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )
    
    # Initialize Neo4j Graph for LangChain
    try:
        app.state.neo4j_graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Neo4jGraph: {e}")
        app.state.neo4j_graph = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down GitConnect API...")
    if hasattr(app.state, "graph_manager"):
        app.state.graph_manager.close()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="GitConnect",
    description="Code dependency visualization and impact analysis using GraphRAG",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class IngestRequest(BaseModel):
    """Request model for repository ingestion."""
    repo_url: str = Field(..., description="GitHub repository URL")
    clear_existing: bool = Field(
        False, description="Clear existing graph before ingestion"
    )

    @pydantic.validator("repo_url")
    def validate_github_url(cls, v):
        if "github.com" not in v:
            raise ValueError("Must be a valid GitHub URL")
        return v


class IngestResponse(BaseModel):
    """Response model for repository ingestion."""
    status: str
    message: str
    stats: Optional[dict] = None




class AnalyzeRequest(BaseModel):
    """Request model for impact analysis."""
    query: str = Field(
        ..., 
        description="Natural language query about code dependencies",
        examples=["What breaks if I delete UserAuth?", "What functions call the database?"]
    )
    use_vector_search: bool = Field(
        True, description="Use vector search for semantic matching"
    )
    max_depth: int = Field(
        3, ge=1, le=10, description="Maximum graph traversal depth"
    )


class AnalyzeResponse(BaseModel):
    """Response model for impact analysis."""
    query: str
    answer: str
    cypher_query: Optional[str] = None
    relevant_entities: list[dict] = []
    dependencies: list[dict] = []
    truncation_warning: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    neo4j_connected: bool
    graph_stats: Optional[dict] = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "Welcome to GitConnect API. Visit /docs for API documentation."}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and database connection status."""
    graph_manager: GraphManager = app.state.graph_manager
    
    try:
        # Test connection
        stats = graph_manager.get_graph_stats()
        return HealthResponse(
            status="healthy",
            neo4j_connected=True,
            graph_stats=stats,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            neo4j_connected=False,
            graph_stats=None,
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_repository(request: IngestRequest, background_tasks: BackgroundTasks):
    """Clone and parse a GitHub repository into the knowledge graph.
    
    This endpoint:
    1. Clones the repository from GitHub
    2. Parses Python and JavaScript files using Tree-sitter
    3. Extracts classes, functions, imports, and call relationships
    4. Stores the code structure in Neo4j
    """
    graph_manager: GraphManager = app.state.graph_manager
    
    try:
        # Clear existing graph if requested
        if request.clear_existing:
            logger.info("Clearing existing graph...")
            graph_manager.clear_graph()
        
        # Clone repository
        logger.info(f"Cloning repository: {request.repo_url}")
        
        # Extract repo name
        if "github.com/" in request.repo_url:
            parts = request.repo_url.split("github.com/")[-1].replace(".git", "").split("/")
            repo_name = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        else:
            repo_name = request.repo_url.split("/")[-1].replace(".git", "")
            
        # Define blocking task wrapper
        def process_repository(url: str, name: str):
            with GitHubFetcher() as fetcher:
                repo_path = fetcher.clone_repository(url)
                
                # Parse repository
                logger.info(f"Parsing repository at: {repo_path}")
                result = parse_repository(repo_path)
                
                # Ingest into graph
                logger.info(f"Ingesting parse results for {name} into Neo4j...")
                return graph_manager.ingest_parse_result(result, name)

        # Run in thread pool to avoid blocking main loop
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(
            None, process_repository, request.repo_url, repo_name
        )
        
        return IngestResponse(
            status="success",
            message=f"Successfully ingested repository: {repo_name}",
            stats=stats,
        )
        
    except GitHubFetchError as e:
        logger.error(f"GitHub fetch error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except GraphConnectionError as e:
        logger.error(f"Graph connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    except Exception as e:
        import traceback
        with open("ingest_error.log", "w") as f:
            f.write(traceback.format_exc())
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.delete("/repos/{repo_name:path}")
async def delete_repository(repo_name: str):
    """Delete a repository from the knowledge graph."""
    graph_manager: GraphManager = app.state.graph_manager
    try:
        graph_manager.delete_repository(repo_name)
        return {"status": "success", "message": f"Repository '{repo_name}' deleted"}
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_impact(request: AnalyzeRequest):
    """Analyze code dependencies using natural language queries.
    
    This endpoint implements a hybrid GraphRAG approach:
    1. Vector search: Find semantically relevant code entities
    2. Graph traversal: Explore structural dependencies
    3. Cypher generation: Convert natural language to graph queries
    4. LLM summarization: Synthesize findings into actionable insights
    """
    graph_manager: GraphManager = app.state.graph_manager
    embeddings: OpenAIEmbeddings = app.state.embeddings
    llm: ChatOpenAI = app.state.llm
    neo4j_graph: Optional[Neo4jGraph] = app.state.neo4j_graph
    
    try:
        relevant_entities = []
        dependencies = []
        cypher_query = None
        
        # Step 1: Vector search for relevant entities (if enabled)
        if request.use_vector_search:
            try:
                query_embedding = embeddings.embed_query(request.query)
                vector_results = graph_manager.vector_search(
                    query_embedding, 
                    node_type="Function",
                    top_k=5,
                )
                relevant_entities.extend(vector_results)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Step 2: Extract entity names from query for direct search
        # Look for capitalized words (Classes/Functions) or words with file extensions or snake_case
        import re
        # Matches: CapitalizedWords, words.with.extensions, or snake_case_words
        entity_patterns = re.findall(r'\b([a-zA-Z0-9_\.]+)\b', request.query)
        # Filter out common stopwords if needed, but for now just search all candidates
        # that look "interesting" (len > 3)
        potential_entities = {p for p in entity_patterns if len(p) > 3}
        
        for entity_name in potential_entities:
            # Search for functions
            funcs = graph_manager.find_function_by_name(entity_name)
            relevant_entities.extend([{"type": "Function", **f} for f in funcs])
            
            # Search for classes
            classes = graph_manager.find_class_by_name(entity_name)
            relevant_entities.extend([{"type": "Class", **c} for c in classes])

            # Search for files
            files = graph_manager.find_file_by_name(entity_name)
            relevant_entities.extend([{"type": "File", **f} for f in files])
            
            # Find dependents (what would break)
            deps = graph_manager.find_dependents(entity_name, max_depth=request.max_depth)
            dependencies.extend(deps)
        
        # Step 3: Use GraphCypherQAChain for natural language to Cypher
        chain_result = None
        if neo4j_graph:
            try:
                # Custom prompt to improve Cypher generation
                CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Important: 
- When searching for files by name, always use `ENDS WITH` or `CONTAINS` as the stored path is a full relative path (e.g. 'src/module/file.py'). Example: `f.path ENDS WITH "adapters.py"`
- `File` nodes do NOT have a `functions` property. Use `(f:File)-[:CONTAINS]->(fn:Function)` to find functions in a file.
- `File` nodes do NOT have a `classes` property. Use `(f:File)-[:CONTAINS]->(c:Class)` to find classes in a file.
- Always use case-insensitive matching if possible.

The question is:
{question}"""

                from langchain_core.prompts import PromptTemplate
                cypher_prompt = PromptTemplate(
                    input_variables=["schema", "question"], 
                    template=CYPHER_GENERATION_TEMPLATE
                )

                cypher_chain = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=neo4j_graph,
                    verbose=True,
                    return_intermediate_steps=True,
                    allow_dangerous_requests=True,
                    cypher_prompt=cypher_prompt,
                    top_k=100,
                )
                
                chain_result = cypher_chain.invoke({"query": request.query})
                intermediate_steps = chain_result.get("intermediate_steps", [])
                if intermediate_steps:
                    cypher_query = intermediate_steps[0].get("query")
                    # Check for truncation in context (step 1)
                    if len(intermediate_steps) > 1:
                        context = intermediate_steps[1].get("context", [])
                        if len(context) >= 100:  # Matches top_k=100
                            truncation_warning = f"Results were truncated to {len(context)} items. Some data may be missing."
                
            except Exception as e:
                logger.warning(f"GraphCypherQAChain failed: {e}")
        
        # Step 4: Generate summary using LLM
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful code analysis assistant.
            
Your task is to answer the user's question based on the provided graph data, entity searches, and dependency analysis.

Resources provided:
1. Relevant Entities: Code entities found via vector/keyword search.
2. Dependencies: Upstream/downstream dependencies found via graph traversal.
3. Graph Query Result: The result of a direct Cypher query generated from the user's natural language question.

Instructions:
- If the user asks a specific structural question (e.g., "What functions are in file X?"), answer it DIRECTLY using the Graph Query Result or Relevant Entities. List the specific names found.
- If the user asks about impact or dependencies (e.g., "What breaks if I change X?"), provide a comprehensive impact analysis (Directly Affected, Chain of Dependencies, Risk Areas).
- Be flexible. Do not force an "Impact Analysis" structure if it doesn't fit the question.
- If the Graph Query Result is empty, rely on Relevant Entities and Dependencies.
- If no information is found, admit it clearly."""),
            ("user", """Query: {query}

Relevant Entities Found:
{entities}

Dependencies (what depends on these entities):
{dependencies}

Graph Query Result:
{chain_result}

Please provide the answer.""")
        ])
        
        # Format the data for the prompt
        entities_str = "\n".join([
            f"- {e.get('type', 'Unknown')}: {e.get('name', 'N/A')} in {e.get('file_path', 'N/A')}"
            for e in relevant_entities[:10]  # Limit to avoid token overflow
        ]) or "No entities found"
        
        deps_str = "\n".join([
            f"- {d.get('type', 'Unknown')}: {d.get('name', 'N/A')} (line {d.get('line', 'N/A')})"
            for d in dependencies[:20]  # Limit to avoid token overflow
        ]) or "No dependencies found"
        
        chain_result_str = str(chain_result.get("result", "")) if chain_result else "N/A"
        
        # Generate summary
        if 'truncation_warning' in locals() and truncation_warning:
             chain_result_str += f"\n\nWARNING: {truncation_warning}"

        messages = summary_prompt.format_messages(
            query=request.query,
            entities=entities_str,
            dependencies=deps_str,
            chain_result=chain_result_str,
        )
        
        summary = llm.invoke(messages)
        
        return AnalyzeResponse(
            query=request.query,
            answer=summary.content,
            cypher_query=cypher_query,
            relevant_entities=relevant_entities[:10],
            dependencies=dependencies[:20],
            truncation_warning=truncation_warning if 'truncation_warning' in locals() else None,
        )
        
    except GraphConnectionError as e:
        logger.error(f"Graph connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
    except Exception as e:
        import traceback
        with open("analyze_error.log", "w") as f:
            f.write(traceback.format_exc())
            if cypher_query:
                f.write(f"\n\nCypher Query: {cypher_query}")
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.get("/stats")
async def get_graph_stats():
    """Get current graph statistics."""
    graph_manager: GraphManager = app.state.graph_manager
    
    try:
        stats = graph_manager.get_graph_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def execute_cypher(
    query: str,
    x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret")
):
    """Execute a raw Cypher query (for debugging/advanced use)."""
    settings = get_settings()
    if x_admin_secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
        
    graph_manager: GraphManager = app.state.graph_manager
    
    try:
        result = graph_manager.execute_cypher(query)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
