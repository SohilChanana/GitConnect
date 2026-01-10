"""Pydantic models for code entities and relationships.

Defines the node and edge types extracted from source code:
- Nodes: File, Class, Function, Module
- Edges: IMPORTS, CALLS, DEFINES
"""

from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Node Models
# =============================================================================


class FileNode(BaseModel):
    """Represents a source code file."""

    path: str = Field(..., description="Absolute or relative path to the file")
    language: str = Field(..., description="Programming language (python, javascript)")
    
    @property
    def node_id(self) -> str:
        """Unique identifier for the file node."""
        return f"file:{self.path}"


class ClassNode(BaseModel):
    """Represents a class definition."""

    name: str = Field(..., description="Class name")
    file_path: str = Field(..., description="Path to the file containing the class")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    docstring: Optional[str] = Field(None, description="Class docstring if present")
    
    @property
    def node_id(self) -> str:
        """Unique identifier for the class node."""
        return f"class:{self.file_path}:{self.name}"
    
    @property
    def qualified_name(self) -> str:
        """Fully qualified name including file path."""
        return f"{self.file_path}::{self.name}"


class FunctionNode(BaseModel):
    """Represents a function or method definition."""

    name: str = Field(..., description="Function name")
    file_path: str = Field(..., description="Path to the file containing the function")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    parent_class: Optional[str] = Field(
        None, description="Parent class name if this is a method"
    )
    docstring: Optional[str] = Field(None, description="Function docstring if present")
    parameters: list[str] = Field(
        default_factory=list, description="List of parameter names"
    )
    
    @property
    def node_id(self) -> str:
        """Unique identifier for the function node."""
        if self.parent_class:
            return f"function:{self.file_path}:{self.parent_class}.{self.name}"
        return f"function:{self.file_path}:{self.name}:{self.start_line}"
    
    @property
    def qualified_name(self) -> str:
        """Fully qualified name including file path and parent class."""
        if self.parent_class:
            return f"{self.file_path}::{self.parent_class}.{self.name}"
        return f"{self.file_path}::{self.name}"
    
    @property
    def is_method(self) -> bool:
        """Check if this function is a method (has parent class)."""
        return self.parent_class is not None


class ModuleNode(BaseModel):
    """Represents an imported module."""

    name: str = Field(..., description="Module name (e.g., 'os', 'flask.app')")
    is_external: bool = Field(
        True, description="Whether this is an external/third-party module"
    )
    
    @property
    def node_id(self) -> str:
        """Unique identifier for the module node."""
        return f"module:{self.name}"


# =============================================================================
# Edge Models
# =============================================================================


class ImportEdge(BaseModel):
    """Represents an import relationship: (File)-[:IMPORTS]->(Module)"""

    source_file: str = Field(..., description="Path to the file doing the import")
    target_module: str = Field(..., description="Name of the imported module")
    imported_names: list[str] = Field(
        default_factory=list,
        description="Specific names imported (for 'from X import Y' statements)",
    )
    line_number: int = Field(..., ge=1, description="Line number of the import")
    is_relative: bool = Field(False, description="Whether this is a relative import")


class CallEdge(BaseModel):
    """Represents a function call: (Function)-[:CALLS]->(Function)"""

    caller_file: str = Field(..., description="File containing the caller")
    caller_name: str = Field(..., description="Name of the calling function")
    caller_class: Optional[str] = Field(
        None, description="Parent class of caller if method"
    )
    callee_name: str = Field(..., description="Name of the called function")
    callee_class: Optional[str] = Field(
        None, description="Parent class of callee if method call"
    )
    line_number: int = Field(..., ge=1, description="Line number of the call")
    
    @property
    def caller_qualified_name(self) -> str:
        """Qualified name of the caller."""
        if self.caller_class:
            return f"{self.caller_class}.{self.caller_name}"
        return self.caller_name
    
    @property
    def callee_qualified_name(self) -> str:
        """Qualified name of the callee."""
        if self.callee_class:
            return f"{self.callee_class}.{self.callee_name}"
        return self.callee_name


class DefinesEdge(BaseModel):
    """Represents a class defining a method: (Class)-[:DEFINES]->(Function)"""

    class_name: str = Field(..., description="Name of the class")
    function_name: str = Field(..., description="Name of the method")
    file_path: str = Field(..., description="File containing the class")


# =============================================================================
# Composite Result Models
# =============================================================================


class ParseResult(BaseModel):
    """Result of parsing a repository or file."""

    files: list[FileNode] = Field(default_factory=list)
    classes: list[ClassNode] = Field(default_factory=list)
    functions: list[FunctionNode] = Field(default_factory=list)
    modules: list[ModuleNode] = Field(default_factory=list)
    imports: list[ImportEdge] = Field(default_factory=list)
    calls: list[CallEdge] = Field(default_factory=list)
    defines: list[DefinesEdge] = Field(default_factory=list)
    
    @property
    def node_count(self) -> int:
        """Total number of nodes."""
        return len(self.files) + len(self.classes) + len(self.functions) + len(self.modules)
    
    @property
    def edge_count(self) -> int:
        """Total number of edges."""
        return len(self.imports) + len(self.calls) + len(self.defines)
    
    def merge(self, other: "ParseResult") -> "ParseResult":
        """Merge another ParseResult into this one."""
        return ParseResult(
            files=self.files + other.files,
            classes=self.classes + other.classes,
            functions=self.functions + other.functions,
            modules=self.modules + other.modules,
            imports=self.imports + other.imports,
            calls=self.calls + other.calls,
            defines=self.defines + other.defines,
        )
