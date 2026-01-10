"""Tests for the Tree-sitter AST parser."""

import tempfile
from pathlib import Path

import pytest

from src.parser import CodeParser, parse_repository
from src.models.entities import ParseResult


class TestCodeParserPython:
    """Tests for Python parsing."""

    def test_parse_simple_function(self):
        """Test parsing a simple Python function."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
def hello(name):
    """Say hello."""
    print(f"Hello, {name}!")
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        assert len(result.files) == 1
        assert result.files[0].language == "python"
        assert len(result.functions) == 1
        assert result.functions[0].name == "hello"
        assert result.functions[0].parameters == ["name"]
        assert result.functions[0].parent_class is None

    def test_parse_class_with_methods(self):
        """Test parsing a Python class with methods."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
class Calculator:
    """A simple calculator."""
    
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value
    
    def subtract(self, x):
        self.value -= x
        return self.value
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        assert len(result.classes) == 1
        assert result.classes[0].name == "Calculator"
        
        # Should have 3 methods
        assert len(result.functions) == 3
        method_names = {f.name for f in result.functions}
        assert method_names == {"__init__", "add", "subtract"}
        
        # All methods should have Calculator as parent
        for func in result.functions:
            assert func.parent_class == "Calculator"
        
        # Should have DEFINES edges
        assert len(result.defines) == 3

    def test_parse_imports(self):
        """Test parsing Python imports."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
import os
import sys
from pathlib import Path
from typing import Optional, List
from .local_module import helper
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        # Should have import edges
        assert len(result.imports) >= 4
        
        # Check module names
        module_names = {imp.target_module for imp in result.imports}
        assert "os" in module_names
        assert "sys" in module_names
        assert "pathlib" in module_names
        assert "typing" in module_names

    def test_parse_function_calls(self):
        """Test parsing function calls."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
def process_data(data):
    result = transform(data)
    validate(result)
    return save(result)

def transform(data):
    return data.upper()
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        assert len(result.functions) == 2
        
        # Should have call edges from process_data
        assert len(result.calls) >= 3
        call_names = {c.callee_name for c in result.calls}
        assert "transform" in call_names
        assert "validate" in call_names
        assert "save" in call_names

    def test_parse_line_numbers(self):
        """Test that line numbers are captured correctly."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''# Line 1
# Line 2
def func_at_line_3():
    pass

class ClassAtLine6:
    def method_at_line_7(self):
        pass
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        # Find the function
        func = next(f for f in result.functions if f.name == "func_at_line_3")
        assert func.start_line == 3
        
        # Find the class
        cls = result.classes[0]
        assert cls.start_line == 6


class TestCodeParserJavaScript:
    """Tests for JavaScript parsing."""

    def test_parse_function_declaration(self):
        """Test parsing JavaScript function declarations."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write('''
function greet(name) {
    console.log("Hello, " + name);
}
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        assert len(result.files) == 1
        assert result.files[0].language == "javascript"
        assert len(result.functions) == 1
        assert result.functions[0].name == "greet"
        assert "name" in result.functions[0].parameters

    def test_parse_arrow_functions(self):
        """Test parsing JavaScript arrow functions."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write('''
const add = (a, b) => a + b;
const multiply = (x, y) => {
    return x * y;
};
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        # Should find arrow functions assigned to variables
        func_names = {f.name for f in result.functions}
        assert "add" in func_names or "multiply" in func_names

    def test_parse_class_declaration(self):
        """Test parsing JavaScript class declarations."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write('''
class User {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
    
    setName(name) {
        this.name = name;
    }
}
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        assert len(result.classes) == 1
        assert result.classes[0].name == "User"
        
        # Should have methods
        method_names = {f.name for f in result.functions}
        assert "constructor" in method_names
        assert "getName" in method_names
        assert "setName" in method_names

    def test_parse_es6_imports(self):
        """Test parsing ES6 imports."""
        parser = CodeParser()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write('''
import React from 'react';
import { useState, useEffect } from 'react';
import './styles.css';
import { helper } from '../utils/helper.js';
''')
            f.flush()
            
            result = parser.parse_file(Path(f.name))
        
        assert len(result.imports) >= 3
        
        # Check imported names
        react_import = next(
            (i for i in result.imports if i.target_module == "react"), None
        )
        assert react_import is not None


class TestParseDirectory:
    """Tests for directory parsing."""

    def test_parse_directory(self):
        """Test parsing a directory with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python file
            py_file = Path(tmpdir) / "main.py"
            py_file.write_text('''
def main():
    print("Hello")
''')
            
            # Create JavaScript file
            js_file = Path(tmpdir) / "app.js"
            js_file.write_text('''
function init() {
    console.log("Init");
}
''')
            
            # Create subdirectory with file
            subdir = Path(tmpdir) / "utils"
            subdir.mkdir()
            util_file = subdir / "helper.py"
            util_file.write_text('''
def helper():
    return True
''')
            
            # Parse directory
            result = parse_repository(Path(tmpdir))
        
        assert len(result.files) == 3
        assert result.node_count >= 3  # At least 3 functions
        
        # Check languages
        languages = {f.language for f in result.files}
        assert "python" in languages
        assert "javascript" in languages


class TestParseResult:
    """Tests for ParseResult model."""

    def test_merge_results(self):
        """Test merging two ParseResults."""
        result1 = ParseResult(
            files=[],
            classes=[],
            functions=[],
            modules=[],
            imports=[],
            calls=[],
            defines=[],
        )
        
        result2 = ParseResult(
            files=[],
            classes=[],
            functions=[],
            modules=[],
            imports=[],
            calls=[],
            defines=[],
        )
        
        merged = result1.merge(result2)
        assert isinstance(merged, ParseResult)

    def test_node_count(self):
        """Test node counting."""
        result = ParseResult()
        assert result.node_count == 0
        assert result.edge_count == 0
