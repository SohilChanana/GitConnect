"""Tree-sitter based AST parser for GitConnect.

Parses Python and JavaScript files to extract:
- Nodes: File, Class, Function
- Edges: IMPORTS, CALLS, DEFINES

Uses tree-sitter for accurate AST parsing instead of regex.
"""

import logging
from pathlib import Path
from typing import Optional

import tree_sitter_languages

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


class CodeParser:
    """Parses source code using Tree-sitter AST."""

    # Map file extensions to language names
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mjs": "javascript",
    }

    def __init__(self):
        """Initialize the parser with tree-sitter languages."""
        self._parsers = {}

    def _get_parser(self, language: str):
        """Get or create a parser for the given language."""
        if language not in self._parsers:
            try:
                self._parsers[language] = tree_sitter_languages.get_parser(language)
            except Exception as e:
                logger.error(f"Failed to load parser for {language}: {e}")
                raise
        return self._parsers[language]

    def _get_language(self, file_path: Path) -> Optional[str]:
        """Determine language from file extension."""
        return self.LANGUAGE_MAP.get(file_path.suffix.lower())

    def _get_node_text(self, node, source_bytes: bytes) -> str:
        """Extract text content from a tree-sitter node."""
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """Extract docstring from a class or function node."""
        # Look for first child that is a string/expression_statement containing a string
        for child in node.children:
            if child.type == "expression_statement":
                for subchild in child.children:
                    if subchild.type == "string":
                        text = self._get_node_text(subchild, source_bytes)
                        # Strip quotes
                        if text.startswith('"""') or text.startswith("'''"):
                            return text[3:-3].strip()
                        elif text.startswith('"') or text.startswith("'"):
                            return text[1:-1].strip()
            elif child.type == "block":
                # Recurse into block
                return self._extract_docstring(child, source_bytes)
        return None

    # =========================================================================
    # Python Parsing
    # =========================================================================

    def _parse_python(
        self, file_path: Path, source_bytes: bytes, relative_path: str
    ) -> ParseResult:
        """Parse a Python file."""
        parser = self._get_parser("python")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        result = ParseResult()
        result.files.append(FileNode(path=relative_path, language="python"))

        # Track current scope for nested parsing
        self._parse_python_node(
            root, source_bytes, relative_path, result, parent_class=None
        )

        return result

    def _parse_python_node(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        parent_class: Optional[str] = None,
        current_function: Optional[str] = None,
    ) -> None:
        """Recursively parse Python AST nodes."""
        
        for child in node.children:
            if child.type == "class_definition":
                self._parse_python_class(child, source_bytes, file_path, result)
            
            elif child.type == "function_definition":
                self._parse_python_function(
                    child, source_bytes, file_path, result, parent_class
                )
            
            elif child.type == "import_statement":
                self._parse_python_import(child, source_bytes, file_path, result)
            
            elif child.type == "import_from_statement":
                self._parse_python_from_import(child, source_bytes, file_path, result)
            
            elif child.type == "call":
                self._parse_python_call(
                    child, source_bytes, file_path, result, 
                    current_function or parent_class
                )
            
            # Recurse into compound statements
            elif child.type in {"block", "if_statement", "for_statement", 
                               "while_statement", "try_statement", "with_statement"}:
                self._parse_python_node(
                    child, source_bytes, file_path, result, 
                    parent_class, current_function
                )

    def _parse_python_class(
        self, node, source_bytes: bytes, file_path: str, result: ParseResult
    ) -> None:
        """Parse a Python class definition."""
        # Find class name
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break

        if not name_node:
            return

        class_name = self._get_node_text(name_node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)

        class_node = ClassNode(
            name=class_name,
            file_path=file_path,
            start_line=node.start_point[0] + 1,  # Convert to 1-indexed
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            content=self._get_node_text(node, source_bytes),
        )
        result.classes.append(class_node)

        # Parse class body for methods
        for child in node.children:
            if child.type == "block":
                self._parse_python_node(
                    child, source_bytes, file_path, result, parent_class=class_name
                )

    def _parse_python_function(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        parent_class: Optional[str],
    ) -> None:
        """Parse a Python function definition."""
        # Find function name
        name_node = None
        params = []
        
        for child in node.children:
            if child.type == "identifier":
                name_node = child
            elif child.type == "parameters":
                # Extract parameter names
                for param in child.children:
                    if param.type == "identifier":
                        params.append(self._get_node_text(param, source_bytes))
                    elif param.type in {"typed_parameter", "default_parameter", 
                                        "typed_default_parameter"}:
                        for subchild in param.children:
                            if subchild.type == "identifier":
                                params.append(self._get_node_text(subchild, source_bytes))
                                break

        if not name_node:
            return

        func_name = self._get_node_text(name_node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)

        func_node = FunctionNode(
            name=func_name,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent_class=parent_class,
            docstring=docstring,
            content=self._get_node_text(node, source_bytes),
            parameters=params,
        )
        result.functions.append(func_node)

        # Create DEFINES edge if this is a method
        if parent_class:
            result.defines.append(
                DefinesEdge(
                    class_name=parent_class,
                    function_name=func_name,
                    file_path=file_path,
                )
            )

        # Parse function body for calls
        for child in node.children:
            if child.type == "block":
                self._parse_python_node(
                    child, source_bytes, file_path, result,
                    parent_class, current_function=func_name
                )

    def _parse_python_import(
        self, node, source_bytes: bytes, file_path: str, result: ParseResult
    ) -> None:
        """Parse a Python import statement (import X)."""
        for child in node.children:
            if child.type == "dotted_name":
                module_name = self._get_node_text(child, source_bytes)
                result.imports.append(
                    ImportEdge(
                        source_file=file_path,
                        target_module=module_name,
                        line_number=node.start_point[0] + 1,
                    )
                )
                # Add module node
                if not any(m.name == module_name for m in result.modules):
                    result.modules.append(ModuleNode(name=module_name))
            
            elif child.type == "aliased_import":
                for subchild in child.children:
                    if subchild.type == "dotted_name":
                        module_name = self._get_node_text(subchild, source_bytes)
                        result.imports.append(
                            ImportEdge(
                                source_file=file_path,
                                target_module=module_name,
                                line_number=node.start_point[0] + 1,
                            )
                        )
                        if not any(m.name == module_name for m in result.modules):
                            result.modules.append(ModuleNode(name=module_name))
                        break

    def _parse_python_from_import(
        self, node, source_bytes: bytes, file_path: str, result: ParseResult
    ) -> None:
        """Parse a Python from-import statement (from X import Y)."""
        module_name = None
        imported_names = []
        is_relative = False

        for child in node.children:
            if child.type == "dotted_name":
                module_name = self._get_node_text(child, source_bytes)
            elif child.type == "relative_import":
                is_relative = True
                for subchild in child.children:
                    if subchild.type == "dotted_name":
                        module_name = self._get_node_text(subchild, source_bytes)
                        break
            elif child.type == "import_prefix":
                is_relative = True
            elif child.type in {"identifier", "dotted_name"} and module_name:
                imported_names.append(self._get_node_text(child, source_bytes))
            elif child.type == "aliased_import":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        imported_names.append(self._get_node_text(subchild, source_bytes))
                        break

        if module_name or is_relative:
            target = module_name or "."
            result.imports.append(
                ImportEdge(
                    source_file=file_path,
                    target_module=target,
                    imported_names=imported_names,
                    line_number=node.start_point[0] + 1,
                    is_relative=is_relative,
                )
            )
            if module_name and not any(m.name == module_name for m in result.modules):
                result.modules.append(ModuleNode(name=module_name, is_external=not is_relative))

    def _parse_python_call(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        caller: Optional[str],
    ) -> None:
        """Parse a Python function call."""
        if not caller:
            return

        # Get the function being called
        callee_name = None
        callee_class = None

        for child in node.children:
            if child.type == "identifier":
                callee_name = self._get_node_text(child, source_bytes)
            elif child.type == "attribute":
                # Method call: obj.method()
                parts = []
                for subchild in child.children:
                    if subchild.type == "identifier":
                        parts.append(self._get_node_text(subchild, source_bytes))
                    elif subchild.type == "attribute":
                        # Nested attribute access
                        for nested in subchild.children:
                            if nested.type == "identifier":
                                parts.append(self._get_node_text(nested, source_bytes))
                
                if len(parts) >= 2:
                    callee_class = parts[-2] if len(parts) > 1 else None
                    callee_name = parts[-1]
                elif parts:
                    callee_name = parts[-1]

        if callee_name:
            result.calls.append(
                CallEdge(
                    caller_file=file_path,
                    caller_name=caller,
                    callee_name=callee_name,
                    callee_class=callee_class,
                    line_number=node.start_point[0] + 1,
                )
            )

    # =========================================================================
    # JavaScript Parsing
    # =========================================================================

    def _parse_javascript(
        self, file_path: Path, source_bytes: bytes, relative_path: str, language: str = "javascript"
    ) -> ParseResult:
        """Parse a JavaScript/TypeScript file."""
        parser = self._get_parser(language)
        tree = parser.parse(source_bytes)
        root = tree.root_node

        result = ParseResult()
        result.files.append(FileNode(path=relative_path, language="javascript"))

        self._parse_js_node(root, source_bytes, relative_path, result)

        return result

    def _parse_js_node(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        parent_class: Optional[str] = None,
        current_function: Optional[str] = None,
    ) -> None:
        """Recursively parse JavaScript AST nodes."""
        
        for child in node.children:
            if child.type == "class_declaration":
                self._parse_js_class(child, source_bytes, file_path, result)
            
            elif child.type == "function_declaration":
                self._parse_js_function(child, source_bytes, file_path, result, parent_class)
            
            elif child.type == "method_definition":
                self._parse_js_method(child, source_bytes, file_path, result, parent_class)
            
            elif child.type in {"arrow_function", "function_expression"}:
                # Handle anonymous/arrow functions when assigned
                pass  # These are handled by variable declarations
            
            elif child.type == "variable_declaration":
                self._parse_js_variable_declaration(
                    child, source_bytes, file_path, result, parent_class
                )
            
            elif child.type == "import_statement":
                self._parse_js_import(child, source_bytes, file_path, result)
            
            elif child.type == "call_expression":
                self._parse_js_call(
                    child, source_bytes, file_path, result, current_function or parent_class
                )
            
            # Recurse into compound statements
            elif child.type in {"statement_block", "if_statement", "for_statement",
                               "while_statement", "try_statement", "program", 
                               "export_statement", "lexical_declaration"}:
                self._parse_js_node(
                    child, source_bytes, file_path, result,
                    parent_class, current_function
                )

    def _parse_js_class(
        self, node, source_bytes: bytes, file_path: str, result: ParseResult
    ) -> None:
        """Parse a JavaScript class declaration."""
        class_name = None
        
        for child in node.children:
            if child.type == "identifier":
                class_name = self._get_node_text(child, source_bytes)
                break

        if not class_name:
            return

        class_node = ClassNode(
            name=class_name,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            content=self._get_node_text(node, source_bytes),
        )
        result.classes.append(class_node)

        # Parse class body
        for child in node.children:
            if child.type == "class_body":
                self._parse_js_node(
                    child, source_bytes, file_path, result, parent_class=class_name
                )

    def _parse_js_function(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        parent_class: Optional[str],
    ) -> None:
        """Parse a JavaScript function declaration."""
        func_name = None
        params = []

        for child in node.children:
            if child.type == "identifier":
                func_name = self._get_node_text(child, source_bytes)
            elif child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "identifier":
                        params.append(self._get_node_text(param, source_bytes))
                    elif param.type in {"assignment_pattern", "rest_pattern"}:
                        for subchild in param.children:
                            if subchild.type == "identifier":
                                params.append(self._get_node_text(subchild, source_bytes))
                                break

        if not func_name:
            return

        func_node = FunctionNode(
            name=func_name,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent_class=parent_class,
            content=self._get_node_text(node, source_bytes),
            parameters=params,
        )
        result.functions.append(func_node)

        # Parse function body for calls
        for child in node.children:
            if child.type == "statement_block":
                self._parse_js_node(
                    child, source_bytes, file_path, result,
                    parent_class, current_function=func_name
                )

    def _parse_js_method(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        parent_class: Optional[str],
    ) -> None:
        """Parse a JavaScript method definition."""
        method_name = None
        params = []

        for child in node.children:
            if child.type == "property_identifier":
                method_name = self._get_node_text(child, source_bytes)
            elif child.type == "formal_parameters":
                for param in child.children:
                    if param.type == "identifier":
                        params.append(self._get_node_text(param, source_bytes))

        if not method_name:
            return

        func_node = FunctionNode(
            name=method_name,
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            parent_class=parent_class,
            content=self._get_node_text(node, source_bytes),
            parameters=params,
        )
        result.functions.append(func_node)

        # Create DEFINES edge
        if parent_class:
            result.defines.append(
                DefinesEdge(
                    class_name=parent_class,
                    function_name=method_name,
                    file_path=file_path,
                )
            )

        # Parse method body
        for child in node.children:
            if child.type == "statement_block":
                self._parse_js_node(
                    child, source_bytes, file_path, result,
                    parent_class, current_function=method_name
                )

    def _parse_js_variable_declaration(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        parent_class: Optional[str],
    ) -> None:
        """Parse variable declarations that might contain arrow functions."""
        for child in node.children:
            if child.type == "variable_declarator":
                var_name = None
                is_function = False
                
                for subchild in child.children:
                    if subchild.type == "identifier":
                        var_name = self._get_node_text(subchild, source_bytes)
                    elif subchild.type in {"arrow_function", "function_expression"}:
                        is_function = True
                        
                        # Extract parameters
                        params = []
                        for func_child in subchild.children:
                            if func_child.type == "formal_parameters":
                                for param in func_child.children:
                                    if param.type == "identifier":
                                        params.append(self._get_node_text(param, source_bytes))
                            elif func_child.type == "identifier":
                                # Single param arrow function: x => x + 1
                                params.append(self._get_node_text(func_child, source_bytes))
                        
                        if var_name:
                            func_node = FunctionNode(
                                name=var_name,
                                file_path=file_path,
                                start_line=subchild.start_point[0] + 1,
                                end_line=subchild.end_point[0] + 1,
                                parent_class=parent_class,
                                parameters=params,
                            )
                            result.functions.append(func_node)

    def _parse_js_import(
        self, node, source_bytes: bytes, file_path: str, result: ParseResult
    ) -> None:
        """Parse a JavaScript import statement."""
        module_name = None
        imported_names = []

        for child in node.children:
            if child.type == "string":
                module_name = self._get_node_text(child, source_bytes).strip("'\"")
            elif child.type == "import_clause":
                for subchild in child.children:
                    if subchild.type == "identifier":
                        imported_names.append(self._get_node_text(subchild, source_bytes))
                    elif subchild.type == "named_imports":
                        for import_spec in subchild.children:
                            if import_spec.type == "import_specifier":
                                for name_node in import_spec.children:
                                    if name_node.type == "identifier":
                                        imported_names.append(
                                            self._get_node_text(name_node, source_bytes)
                                        )
                                        break

        if module_name:
            is_relative = module_name.startswith(".")
            result.imports.append(
                ImportEdge(
                    source_file=file_path,
                    target_module=module_name,
                    imported_names=imported_names,
                    line_number=node.start_point[0] + 1,
                    is_relative=is_relative,
                )
            )
            if not any(m.name == module_name for m in result.modules):
                result.modules.append(ModuleNode(name=module_name, is_external=not is_relative))

    def _parse_js_call(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        result: ParseResult,
        caller: Optional[str],
    ) -> None:
        """Parse a JavaScript function call."""
        if not caller:
            return

        callee_name = None
        callee_class = None

        for child in node.children:
            if child.type == "identifier":
                callee_name = self._get_node_text(child, source_bytes)
            elif child.type == "member_expression":
                # Method call: obj.method()
                parts = []
                self._extract_member_parts(child, source_bytes, parts)
                if len(parts) >= 2:
                    callee_class = parts[-2]
                    callee_name = parts[-1]
                elif parts:
                    callee_name = parts[-1]

        if callee_name:
            result.calls.append(
                CallEdge(
                    caller_file=file_path,
                    caller_name=caller,
                    callee_name=callee_name,
                    callee_class=callee_class,
                    line_number=node.start_point[0] + 1,
                )
            )

    def _extract_member_parts(
        self, node, source_bytes: bytes, parts: list[str]
    ) -> None:
        """Recursively extract parts from a member expression."""
        for child in node.children:
            if child.type == "identifier":
                parts.append(self._get_node_text(child, source_bytes))
            elif child.type == "property_identifier":
                parts.append(self._get_node_text(child, source_bytes))
            elif child.type == "member_expression":
                self._extract_member_parts(child, source_bytes, parts)

    # =========================================================================
    # Public API
    # =========================================================================

    def parse_file(self, file_path: Path, base_path: Optional[Path] = None) -> ParseResult:
        """Parse a single source file.
        
        Args:
            file_path: Path to the source file.
            base_path: Optional base path for computing relative paths.
            
        Returns:
            ParseResult containing extracted nodes and edges.
        """
        language = self._get_language(file_path)
        if not language:
            logger.warning(f"Unsupported file type: {file_path}")
            return ParseResult()

        try:
            source_bytes = file_path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return ParseResult()

        # Compute relative path
        if base_path:
            try:
                relative_path = str(file_path.relative_to(base_path))
            except ValueError:
                relative_path = str(file_path)
        else:
            relative_path = str(file_path)

        # Normalize path separators
        relative_path = relative_path.replace("\\", "/")

        logger.info(f"Parsing {relative_path} ({language})")

        if language == "python":
            return self._parse_python(file_path, source_bytes, relative_path)
        elif language in {"javascript", "typescript"}:
            return self._parse_javascript(file_path, source_bytes, relative_path, language)
        else:
            return ParseResult()

    def parse_directory(self, directory: Path) -> ParseResult:
        """Parse all source files in a directory.
        
        Args:
            directory: Path to the directory to parse.
            
        Returns:
            ParseResult containing all extracted nodes and edges.
        """
        result = ParseResult()
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.LANGUAGE_MAP:
                # Skip common non-source directories
                parts = file_path.parts
                if any(p in {"node_modules", "__pycache__", "venv", "env", ".git", 
                            "dist", "build"} for p in parts):
                    continue
                
                file_result = self.parse_file(file_path, base_path=directory)
                result = result.merge(file_result)

        logger.info(
            f"Parsed {len(result.files)} files, found {result.node_count} nodes "
            f"and {result.edge_count} edges"
        )
        return result


def parse_repository(repo_path: Path) -> ParseResult:
    """Convenience function to parse an entire repository.
    
    Args:
        repo_path: Path to the repository root.
        
    Returns:
        ParseResult containing all extracted nodes and edges.
    """
    parser = CodeParser()
    return parser.parse_directory(repo_path)
