#!/usr/bin/env python3
"""
PyDTNN auto-documentation utility.

This module provides automated docstring generation for the PyDTNN framework
using LLM-based code analysis to ensure consistent documentation standards.
"""

import ast
import os
import time
from pathlib import Path
from warnings import warn

from openai import InternalServerError, OpenAI

config = {
    "model": "gemini-3.1-flash-lite",
    "reasoning_effort": "minimal",
    "temperature": 0.1,
}
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def should_process_file(code: str) -> bool:
    """
    Check if a Python file requires docstring updates.

    Args:
        code: The source code string to analyze.

    Returns:
        True if the module, classes, or functions lack docstrings, False otherwise.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    # Docstring de módulo
    if ast.get_docstring(tree) is None:
        return True

    # Clases y funciones
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is None:
                return True

    return False


def is_valid_python(code: str) -> bool:
    """
    Verify if the provided string is valid Python syntax.

    Args:
        code: The source code string to validate.

    Returns:
        True if the code is syntactically correct, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def generate_file_docstrings(path: Path, code: str) -> str:
    """
    Generate missing docstrings for a file using an LLM.

    Args:
        path: The path to the file being processed.
        code: The original source code.

    Returns:
        The source code updated with generated docstrings.
    """
    prompt = f"""
You are a senior Python engineer.

Task:
Add missing docstrings to this file.

Context:
- Project: PyDTNN (ML framework)
- File path: {path}

STRICT RULES:
- ONLY add docstrings
- Target ONLY:
  - module docstring (top of file)
  - class docstrings
  - function/method docstrings
- DO NOT modify any existing code
- DO NOT change formatting
- DO NOT rename anything
- DO NOT add comments
- DO NOT remove anything
- Preserve exact behavior

Docstring rules:
- Be concise and precise
- No hallucinations

Output rules:
- Return the whole file back
- Return ONLY valid Python code
- No markdown, no explanations

Code:
{code}
"""

    response = client.chat.completions.create(
        **config,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content or code

    content = content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("python"):
            content = content[len("python"):]

    return content


def process_file(path: Path) -> bool:
    """
    Process a single file to add missing docstrings.

    Args:
        path: The path to the file to process.

    Returns:
        True if the file was processed or skipped successfully, False otherwise.
    """
    original = path.read_text()

    # Skip empty or finished files
    if not should_process_file(original):
        return False

    generated = generate_file_docstrings(path, original)

    if not generated:
        return True

    # Validar sintaxis
    if not is_valid_python(generated):
        warn(f"Invalid Python file generated ({path})!", RuntimeWarning)

    # Evitar sobrescribir si no hay cambios
    if generated.strip() == original.strip():
        return True

    path.write_text(generated)
    print(f"Updated {path}")
    return True


def process_project(root: Path, name="*.py", delay: float = 5.0) -> None:
    """
    Recursively process all Python files in a directory.

    Args:
        root: The root directory to search.
        name: The file pattern to match.
        delay: The base delay in seconds between API calls.
    """
    base_delay = delay
    for file in root.rglob(name):
        while True:
            delay = max(base_delay, delay)
            try:
                if process_file(file):
                    delay /= 2
                    time.sleep(delay)
                break
            except InternalServerError as e:
                delay *= 2
                print(f"Error: {e}")
                print(f"Backing off {delay}s")
                time.sleep(delay)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dir>")

    root = Path(sys.argv[1])
    if root.is_dir():
        process_project(root)
    else:
        process_file(root)
