#!/usr/bin/env python3
"""Utility module for automatically managing __all__ exports in PyDTNN source files."""

import ast
from pathlib import Path


def get_names(source: str) -> list[str]:
    """Extracts public function and class names from the provided source code."""
    public_names = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return public_names

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_names.append(node.name)

    return sorted(public_names)


def remove_all(source: str) -> str:
    """Removes existing __all__ definitions from the provided source code."""
    to_remove = set[int]()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    start = node.lineno - 1
                    end = node.end_lineno or start
                    for i in range(start, end):
                        to_remove.add(i)

    return "\n".join([line for i, line in enumerate(source.splitlines()) if i not in to_remove])


def find_insert(source: str) -> int:
    """Determines the optimal line index to insert the __all__ definition."""
    end = len(source.splitlines()) - 1

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return end

    body = list(tree.body)

    # Skip module docstring
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]

    # Skip imports
    idx = 0
    while idx < len(body) and isinstance(body[idx], (ast.Import, ast.ImportFrom)):
        idx += 1

    if idx < len(body):
        return body[idx].lineno - 1

    return end


def add_all(source: str, names: list[str]) -> str:
    """Generates and inserts an __all__ definition into the source code."""
    if not names:
        return source

    all_block = ["__all__ = ("]
    for name in names:
        all_block.append(f'    "{name}",')
    all_block.append(")\n")

    insert_idx = find_insert(source)

    lines = source.splitlines()
    source = "\n".join(lines[:insert_idx] + all_block + lines[insert_idx:])

    return source


def process_file(path: Path, replace: bool = False) -> None:
    """Processes a single file to update its __all__ definition."""
    source = path.read_text()
    clean = remove_all(source)

    if not replace and source != clean:
        print(f"Skipped {path}")
        return

    names = get_names(clean)
    source = add_all(clean, names)
    path.write_text(source)
    print(f"Updated {path}")


def process_project(root: Path, name: str = "*.py", replace: bool = False) -> None:
    """Recursively processes all Python files in the given directory."""
    for path in root.rglob(name):
        process_file(path, replace)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <src>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if root.is_dir():
        process_project(root)
    else:
        process_file(root)
