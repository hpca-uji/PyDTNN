#!/usr/bin/env python3
import ast
import os
import time
from pathlib import Path
from warnings import warn

from openai import OpenAI

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def should_process_file(code: str) -> bool:
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
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def generate_file_docstrings(path: Path, code: str) -> str:
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
- Return ONLY valid Python code
- No markdown, no explanations

Code:
{code}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    content = response.choices[0].message.content or code

    content = content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("python"):
            content = content[len("python"):]

    return content


def process_file(path: Path) -> None:
    original = path.read_text()

    # Skip empty or finished files
    if not should_process_file(original):
        return

    generated = generate_file_docstrings(path, original)

    if not generated:
        return

    # Validar sintaxis
    if not is_valid_python(generated):
        warn("Invalid Python file generated!", RuntimeWarning)

    # Evitar sobrescribir si no hay cambios
    if generated.strip() == original.strip():
        return

    path.write_text(generated)
    print(f"Updated {path}")


def process_project(root: Path, name="*.py", delay: float = 1.0) -> None:
    for file in root.rglob(name):
        process_file(file)
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
