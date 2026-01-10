"""Tests for the Neo4j Graph Manager."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.graph_manager import GraphManager, GraphConnectionError
from src.models.entities import (
    ClassNode,
    FileNode,
    FunctionNode,
    ModuleNode,
    ImportEdge,
    CallEdge,
    DefinesEdge,
    ParseResult,
)


class TestGraphManagerConnection:
    """Tests for connection management."""

    @patch("src.graph_manager.GraphDatabase")
    def test_connect_success(self, mock_graph_db):
        """Test successful connection."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager.connect()
        
        mock_driver.verify_connectivity.assert_called_once()

    @patch("src.graph_manager.GraphDatabase")
    def test_connect_auth_error(self, mock_graph_db):
        """Test connection with auth error."""
        from neo4j.exceptions import AuthError
        
        mock_graph_db.driver.side_effect = AuthError("Invalid credentials")
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="wrong",
            )
            
            manager = GraphManager()
            
            with pytest.raises(GraphConnectionError) as exc_info:
                manager.connect()
            
            assert "Authentication failed" in str(exc_info.value)

    @patch("src.graph_manager.GraphDatabase")
    def test_context_manager(self, mock_graph_db):
        """Test context manager usage."""
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            with GraphManager() as manager:
                pass
        
        mock_driver.close.assert_called_once()


class TestNodeCreation:
    """Tests for node creation."""

    @patch("src.graph_manager.GraphDatabase")
    def test_create_file_node(self, mock_graph_db):
        """Test creating a file node."""
        mock_session = MagicMock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            file_node = FileNode(path="src/main.py", language="python")
            manager.create_file_node(file_node)
        
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MERGE" in call_args[0][0]
        assert call_args[1]["path"] == "src/main.py"

    @patch("src.graph_manager.GraphDatabase")
    def test_create_function_node(self, mock_graph_db):
        """Test creating a function node."""
        mock_session = MagicMock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            func_node = FunctionNode(
                name="process",
                file_path="src/utils.py",
                start_line=10,
                end_line=20,
                parent_class=None,
                parameters=["data", "options"],
            )
            manager.create_function_node(func_node)
        
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert call_args[1]["name"] == "process"
        assert call_args[1]["file_path"] == "src/utils.py"


class TestRelationshipCreation:
    """Tests for relationship creation."""

    @patch("src.graph_manager.GraphDatabase")
    def test_create_import_relationship(self, mock_graph_db):
        """Test creating an import relationship."""
        mock_session = MagicMock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            edge = ImportEdge(
                source_file="src/main.py",
                target_module="flask",
                imported_names=["Flask", "request"],
                line_number=1,
            )
            manager.create_import_relationship(edge)
        
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "IMPORTS" in call_args[0][0]

    @patch("src.graph_manager.GraphDatabase")
    def test_create_calls_relationship(self, mock_graph_db):
        """Test creating a calls relationship."""
        mock_session = MagicMock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            edge = CallEdge(
                caller_file="src/main.py",
                caller_name="main",
                callee_name="process",
                line_number=15,
            )
            manager.create_calls_relationship(edge)
        
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CALLS" in call_args[0][0]


class TestBatchOperations:
    """Tests for batch operations."""

    @patch("src.graph_manager.GraphDatabase")
    def test_ingest_parse_result(self, mock_graph_db):
        """Test ingesting a complete parse result."""
        mock_session = MagicMock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            # Create a parse result with some data
            result = ParseResult(
                files=[FileNode(path="main.py", language="python")],
                classes=[ClassNode(
                    name="MyClass",
                    file_path="main.py",
                    start_line=1,
                    end_line=10,
                )],
                functions=[FunctionNode(
                    name="my_func",
                    file_path="main.py",
                    start_line=12,
                    end_line=15,
                )],
                modules=[ModuleNode(name="os")],
                imports=[ImportEdge(
                    source_file="main.py",
                    target_module="os",
                    line_number=1,
                )],
                calls=[],
                defines=[],
            )
            
            counts = manager.ingest_parse_result(result)
        
        assert counts["files"] == 1
        assert counts["classes"] == 1
        assert counts["functions"] == 1
        assert counts["modules"] == 1
        assert counts["imports"] == 1


class TestQueryOperations:
    """Tests for query operations."""

    @patch("src.graph_manager.GraphDatabase")
    def test_find_function_by_name(self, mock_graph_db):
        """Test finding functions by name."""
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([
            {"name": "process", "file_path": "main.py", "start_line": 10, "parent_class": None}
        ]))
        
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            results = manager.find_function_by_name("process")
        
        assert len(results) == 1
        assert results[0]["name"] == "process"

    @patch("src.graph_manager.GraphDatabase")
    def test_get_graph_stats(self, mock_graph_db):
        """Test getting graph statistics."""
        mock_single = Mock()
        mock_single.return_value = {"count": 5}
        
        mock_result = Mock()
        mock_result.single = mock_single
        
        mock_session = MagicMock()
        mock_session.run.return_value = mock_result
        
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_graph_db.driver.return_value = mock_driver
        
        with patch("src.graph_manager.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="password",
            )
            
            manager = GraphManager()
            manager._driver = mock_driver
            
            stats = manager.get_graph_stats()
        
        # Should have queried for each entity type
        assert "files" in stats
        assert "classes" in stats
        assert "functions" in stats


class TestEntityModels:
    """Tests for entity models."""

    def test_file_node_id(self):
        """Test FileNode unique ID generation."""
        node = FileNode(path="src/main.py", language="python")
        assert node.node_id == "file:src/main.py"

    def test_class_node_qualified_name(self):
        """Test ClassNode qualified name."""
        node = ClassNode(
            name="UserService",
            file_path="src/services/user.py",
            start_line=10,
            end_line=50,
        )
        assert node.qualified_name == "src/services/user.py::UserService"

    def test_function_node_is_method(self):
        """Test FunctionNode method detection."""
        func = FunctionNode(
            name="process",
            file_path="main.py",
            start_line=1,
            end_line=5,
            parent_class=None,
        )
        assert not func.is_method
        
        method = FunctionNode(
            name="process",
            file_path="main.py",
            start_line=1,
            end_line=5,
            parent_class="MyClass",
        )
        assert method.is_method

    def test_parse_result_merge(self):
        """Test ParseResult merging."""
        result1 = ParseResult(
            files=[FileNode(path="a.py", language="python")],
            classes=[],
            functions=[],
            modules=[],
            imports=[],
            calls=[],
            defines=[],
        )
        
        result2 = ParseResult(
            files=[FileNode(path="b.py", language="python")],
            classes=[],
            functions=[],
            modules=[],
            imports=[],
            calls=[],
            defines=[],
        )
        
        merged = result1.merge(result2)
        assert len(merged.files) == 2
