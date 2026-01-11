"""Neo4j Graph Manager for GitConnect.

Handles all Neo4j operations including:
- Connection management
- Schema enforcement (constraints and indexes)
- Node and relationship creation
- Vector search for similarity queries
- Graph traversal for impact analysis
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.config import get_settings
from src.models.entities import (
    CallEdge,
    ClassNode,
    DefinesEdge,
    FileNode,
    FunctionNode,
    ImportEdge,
    ModuleNode,
    ParseResult,
)

logger = logging.getLogger(__name__)


class GraphConnectionError(Exception):
    """Raised when Neo4j connection fails."""
    pass


class GraphManager:
    """Manages Neo4j graph operations for code analysis."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize graph manager with Neo4j credentials.
        
        Args:
            uri: Neo4j connection URI. Defaults to config value.
            username: Neo4j username. Defaults to config value.
            password: Neo4j password. Defaults to config value.
        """
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.username = username or settings.neo4j_username
        self.password = password or settings.neo4j_password
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            # Verify connection
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except AuthError as e:
            raise GraphConnectionError(f"Authentication failed: {e}") from e
        except ServiceUnavailable as e:
            raise GraphConnectionError(f"Neo4j service unavailable: {e}") from e
        except Exception as e:
            raise GraphConnectionError(f"Failed to connect to Neo4j: {e}") from e

    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver, connecting if necessary."""
        if not self._driver:
            self.connect()
        return self._driver

    @contextmanager
    def session(self):
        """Create a session context manager."""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def __enter__(self) -> "GraphManager":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    # =========================================================================
    # Schema Management
    # =========================================================================

    def setup_schema(self) -> None:
        """Set up graph schema with constraints and indexes."""
        constraints = [
            # Drop old constraints (manual cleanup might be needed if naming differs)
            "DROP CONSTRAINT file_path IF EXISTS",
            "DROP CONSTRAINT module_name IF EXISTS",
            "DROP CONSTRAINT class_unique IF EXISTS",
            "DROP CONSTRAINT function_unique IF EXISTS",

            # New composite constraints with repo_name
            "CREATE CONSTRAINT file_unique IF NOT EXISTS FOR (f:File) REQUIRE (f.repo_name, f.path) IS UNIQUE",
            "CREATE CONSTRAINT module_unique IF NOT EXISTS FOR (m:Module) REQUIRE (m.repo_name, m.name) IS UNIQUE",
            "CREATE CONSTRAINT class_unique_repo IF NOT EXISTS FOR (c:Class) REQUIRE (c.repo_name, c.file_path, c.name) IS UNIQUE",
            "CREATE CONSTRAINT function_unique_repo IF NOT EXISTS FOR (fn:Function) REQUIRE (fn.repo_name, fn.file_path, fn.start_line) IS UNIQUE",
        ]
        
        indexes = [
            # Indexes for common queries
            "CREATE INDEX file_language IF NOT EXISTS FOR (f:File) ON (f.language)",
            "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX function_name IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
            "CREATE INDEX function_parent IF NOT EXISTS FOR (fn:Function) ON (fn.parent_class)",
            "CREATE INDEX repo_name IF NOT EXISTS FOR (n:File) ON (n.repo_name)",
        ]

        with self.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
        
        logger.info("Graph schema setup complete")

    def setup_vector_index(self, dimensions: int = 1536) -> None:
        """Set up vector index for embedding-based search.
        
        Args:
            dimensions: Embedding vector dimensions (default: 1536 for OpenAI).
        """
        vector_index_query = f"""
        CREATE VECTOR INDEX function_embeddings IF NOT EXISTS
        FOR (f:Function)
        ON f.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        with self.session() as session:
            try:
                session.run(vector_index_query)
                logger.info("Vector index created for Function nodes")
            except Exception as e:
                logger.warning(f"Vector index may already exist: {e}")

    def clear_graph(self) -> None:
        """Delete all nodes and relationships from the graph."""
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Graph cleared")

    def delete_repository(self, repo_name: str) -> None:
        """Delete all nodes associated with a specific repository."""
        query = """
        MATCH (n)
        WHERE n.repo_name = $repo_name
        DETACH DELETE n
        """
        with self.session() as session:
            session.run(query, repo_name=repo_name)
            logger.info(f"Deleted repository: {repo_name}")

    # =========================================================================
    # Node Creation
    # =========================================================================

    def create_file_node(self, file: FileNode, repo_name: str) -> None:
        """Create a File node."""
        query = """
        MERGE (f:File {repo_name: $repo_name, path: $path})
        SET f.language = $language
        """
        with self.session() as session:
            session.run(query, repo_name=repo_name, path=file.path, language=file.language)

    def create_class_node(self, cls: ClassNode, repo_name: str) -> None:
        """Create a Class node."""
        query = """
        MERGE (c:Class {repo_name: $repo_name, file_path: $file_path, name: $name})
        SET c.start_line = $start_line,
            c.end_line = $end_line,
            c.docstring = $docstring,
            c.qualified_name = $qualified_name
        """
        with self.session() as session:
            session.run(
                query,
                repo_name=repo_name,
                file_path=cls.file_path,
                name=cls.name,
                start_line=cls.start_line,
                end_line=cls.end_line,
                docstring=cls.docstring,
                qualified_name=cls.qualified_name,
            )

    def create_function_node(self, func: FunctionNode, repo_name: str) -> None:
        """Create a Function node."""
        query = """
        MERGE (fn:Function {repo_name: $repo_name, file_path: $file_path, name: $name, start_line: $start_line})
        SET fn.end_line = $end_line,
            fn.parent_class = $parent_class,
            fn.docstring = $docstring,
            fn.parameters = $param_list,
            fn.qualified_name = $qualified_name,
            fn.is_method = $is_method
        """
        with self.session() as session:
            session.run(
                query,
                repo_name=repo_name,
                file_path=func.file_path,
                name=func.name,
                start_line=func.start_line,
                end_line=func.end_line,
                parent_class=func.parent_class,
                docstring=func.docstring,
                param_list=func.parameters,
                qualified_name=func.qualified_name,
                is_method=func.is_method,
            )

    def create_module_node(self, module: ModuleNode, repo_name: str) -> None:
        """Create a Module node."""
        query = """
        MERGE (m:Module {repo_name: $repo_name, name: $name})
        SET m.is_external = $is_external
        """
        with self.session() as session:
            session.run(query, repo_name=repo_name, name=module.name, is_external=module.is_external)

    # =========================================================================
    # Relationship Creation
    # =========================================================================

    def create_import_relationship(self, edge: ImportEdge, repo_name: str) -> None:
        """Create an IMPORTS relationship: (File)-[:IMPORTS]->(Module)."""
        query = """
        MATCH (f:File {repo_name: $repo_name, path: $source_file})
        MERGE (m:Module {repo_name: $repo_name, name: $target_module})
        MERGE (f)-[r:IMPORTS]->(m)
        SET r.line_number = $line_number,
            r.imported_names = $imported_names,
            r.is_relative = $is_relative
        """
        with self.session() as session:
            session.run(
                query,
                repo_name=repo_name,
                source_file=edge.source_file,
                target_module=edge.target_module,
                line_number=edge.line_number,
                imported_names=edge.imported_names,
                is_relative=edge.is_relative,
            )

    def create_calls_relationship(self, edge: CallEdge, repo_name: str) -> None:
        """Create a CALLS relationship: (Function)-[:CALLS]->(Function).
        
        Note: This may not find a target function if it's external or not parsed.
        """
        # First try to find an exact match
        query = """
        MATCH (caller:Function {repo_name: $repo_name, file_path: $caller_file, name: $caller_name})
        OPTIONAL MATCH (callee:Function {repo_name: $repo_name, name: $callee_name})
        WHERE callee.parent_class = $callee_class OR ($callee_class IS NULL AND callee.parent_class IS NULL)
        WITH caller, callee
        WHERE callee IS NOT NULL
        MERGE (caller)-[r:CALLS]->(callee)
        SET r.line_number = $line_number
        """
        with self.session() as session:
            session.run(
                query,
                repo_name=repo_name,
                caller_file=edge.caller_file,
                caller_name=edge.caller_name,
                callee_name=edge.callee_name,
                callee_class=edge.callee_class,
                line_number=edge.line_number,
            )

    def create_defines_relationship(self, edge: DefinesEdge, repo_name: str) -> None:
        """Create a DEFINES relationship: (Class)-[:DEFINES]->(Function)."""
        query = """
        MATCH (c:Class {repo_name: $repo_name, file_path: $file_path, name: $class_name})
        MATCH (fn:Function {repo_name: $repo_name, file_path: $file_path, name: $function_name, parent_class: $class_name})
        MERGE (c)-[r:DEFINES]->(fn)
        """
        with self.session() as session:
            session.run(
                query,
                repo_name=repo_name,
                file_path=edge.file_path,
                class_name=edge.class_name,
                function_name=edge.function_name,
            )

    def create_contains_relationship(self, file_path: str, entity_type: str, entity_name: str, repo_name: str) -> None:
        """Create a CONTAINS relationship: (File)-[:CONTAINS]->(Class|Function)."""
        query = f"""
        MATCH (f:File {{repo_name: $repo_name, path: $file_path}})
        MATCH (e:{entity_type} {{repo_name: $repo_name, file_path: $file_path, name: $entity_name}})
        MERGE (f)-[:CONTAINS]->(e)
        """
        with self.session() as session:
            session.run(
                query,
                repo_name=repo_name,
                file_path=file_path,
                entity_name=entity_name,
            )

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def ingest_parse_result(self, result: ParseResult, repo_name: str) -> dict[str, int]:
        """Ingest a complete parse result into the graph.
        
        Args:
            result: ParseResult from the parser.
            repo_name: Name of the repository to isolate nodes.
            
        Returns:
            Dictionary with counts of created entities.
        """
        counts = {
            "files": 0,
            "classes": 0,
            "functions": 0,
            "modules": 0,
            "imports": 0,
            "calls": 0,
            "defines": 0,
        }

        # Create nodes first
        for file in result.files:
            self.create_file_node(file, repo_name)
            counts["files"] += 1

        for cls in result.classes:
            self.create_class_node(cls, repo_name)
            self.create_contains_relationship(cls.file_path, "Class", cls.name, repo_name)
            counts["classes"] += 1

        for func in result.functions:
            self.create_function_node(func, repo_name)
            self.create_contains_relationship(func.file_path, "Function", func.name, repo_name)
            counts["functions"] += 1

        for module in result.modules:
            self.create_module_node(module, repo_name)
            counts["modules"] += 1

        # Create relationships
        for imp in result.imports:
            self.create_import_relationship(imp, repo_name)
            counts["imports"] += 1

        for call in result.calls:
            self.create_calls_relationship(call, repo_name)
            counts["calls"] += 1

        for defines in result.defines:
            self.create_defines_relationship(defines, repo_name)
            counts["defines"] += 1

        logger.info(f"Ingested: {counts}")
        return counts

    # =========================================================================
    # Query Operations
    # =========================================================================

    def find_function_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find functions by name."""
        query = """
        MATCH (fn:Function)
        WHERE fn.name CONTAINS $name
        RETURN fn.name AS name, fn.file_path AS file_path, 
               fn.start_line AS start_line, fn.end_line AS end_line, fn.parent_class AS parent_class,
               fn.repo_name AS repo_name
        """
        with self.session() as session:
            result = session.run(query, name=name)
            return [dict(record) for record in result]

    def find_class_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find classes by name."""
        query = """
        MATCH (c:Class)
        WHERE c.name CONTAINS $name
        RETURN c.name AS name, c.file_path AS file_path,
               c.start_line AS start_line, c.end_line AS end_line,
               c.repo_name AS repo_name
        """
        with self.session() as session:
            result = session.run(query, name=name)
            return [dict(record) for record in result]

    def find_file_by_name(self, name: str) -> list[dict[str, Any]]:
        """Find files by path/name."""
        query = """
        MATCH (f:File)
        WHERE f.path CONTAINS $name
        RETURN f.path AS file_path, f.language AS language, f.repo_name AS repo_name
        """
        with self.session() as session:
            result = session.run(query, name=name)
            return [dict(record) for record in result]

    def find_dependents(self, entity_name: str, max_depth: int = 3) -> list[dict[str, Any]]:
        """Find all entities that depend on the given entity.
        
        This finds what would break if the entity is deleted.
        """
        query = f"""
        MATCH (target)
        WHERE target.name = $name
        MATCH (dependent)-[:CALLS|IMPORTS*1..{max_depth}]->(target)
        RETURN DISTINCT 
            labels(dependent)[0] AS type,
            dependent.name AS name,
            dependent.file_path AS file_path,
            dependent.start_line AS line
        """
        with self.session() as session:
            result = session.run(query, name=entity_name)
            return [dict(record) for record in result]

    def find_dependencies(self, entity_name: str, max_depth: int = 3) -> list[dict[str, Any]]:
        """Find all entities that the given entity depends on."""
        query = f"""
        MATCH (source)
        WHERE source.name = $name
        MATCH (source)-[:CALLS|IMPORTS*1..{max_depth}]->(dependency)
        RETURN DISTINCT
            labels(dependency)[0] AS type,
            dependency.name AS name,
            dependency.file_path AS file_path
        """
        with self.session() as session:
            result = session.run(query, name=entity_name)
            return [dict(record) for record in result]

    def get_graph_stats(self) -> dict[str, int]:
        """Get statistics about the graph."""
        queries = {
            "files": "MATCH (n:File) RETURN count(n) AS count",
            "classes": "MATCH (n:Class) RETURN count(n) AS count",
            "functions": "MATCH (n:Function) RETURN count(n) AS count",
            "modules": "MATCH (n:Module) RETURN count(n) AS count",
            "imports": "MATCH ()-[r:IMPORTS]->() RETURN count(r) AS count",
            "calls": "MATCH ()-[r:CALLS]->() RETURN count(r) AS count",
            "defines": "MATCH ()-[r:DEFINES]->() RETURN count(r) AS count",
        }
        
        stats = {}
        with self.session() as session:
            for key, query in queries.items():
                result = session.run(query)
                stats[key] = result.single()["count"]
        
        return stats

    def get_current_repo(self) -> Optional[str]:
        """Get the name of the currently ingested repository.
        
        Returns:
            The repository name if one exists, else None.
        """
        query = "MATCH (f:File) RETURN f.repo_name AS repo_name LIMIT 1"
        with self.session() as session:
            result = session.run(query)
            record = result.single()
            return record["repo_name"] if record else None

    # =========================================================================
    # Vector Search (for embeddings)
    # =========================================================================

    def update_embedding(self, node_type: str, name: str, file_path: str, embedding: list[float]) -> None:
        """Update the embedding for a node."""
        query = f"""
        MATCH (n:{node_type} {{name: $name, file_path: $file_path}})
        SET n.embedding = $embedding
        """
        with self.session() as session:
            session.run(query, name=name, file_path=file_path, embedding=embedding)

    def vector_search(
        self,
        embedding: list[float],
        node_type: str = "Function",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            embedding: Query embedding vector.
            node_type: Type of node to search.
            top_k: Number of results to return.
            
        Returns:
            List of matching nodes with similarity scores.
        """
        # Note: This requires the vector index to be set up
        query = f"""
        CALL db.index.vector.queryNodes(
            'function_embeddings',
            $top_k,
            $embedding
        ) YIELD node, score
        RETURN node.name AS name, 
               node.file_path AS file_path,
               node.qualified_name AS qualified_name,
               score
        """
        with self.session() as session:
            try:
                result = session.run(query, top_k=top_k, embedding=embedding)
                return [dict(record) for record in result]
            except Exception as e:
                logger.warning(f"Vector search failed (index may not exist): {e}")
                return []

    def execute_cypher(self, query: str, parameters: Optional[dict] = None) -> list[dict[str, Any]]:
        """Execute an arbitrary Cypher query.
        
        Args:
            query: Cypher query string.
            parameters: Optional query parameters.
            
        Returns:
            List of result records as dictionaries.
        """
        with self.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
