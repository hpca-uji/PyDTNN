#!/usr/bin/env python3
import ast
from pathlib import Path


def get_public_api(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    public_names = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_names.append(node.name)

    return sorted(public_names), tree


def remove_existing_all(lines, tree):
    """Remove any existing __all__ assignment (single or multiline)."""
    to_remove = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    start = node.lineno - 1
                    end = node.end_lineno
                    assert end
                    for i in range(start, end):
                        to_remove.add(i)

    return [line for i, line in enumerate(lines) if i not in to_remove]


def find_insert_position(lines):
    insert_idx = 0



    while insert_idx < len(lines):
        stripped = lines[insert_idx].strip()

        # Skip comments and imports
        if not stripped or stripped.startswith("#") or stripped.startswith("import ") or stripped.startswith("from "):
            insert_idx += 1

        # Skip docstring
        elif lines and lines[insert_idx].lstrip().startswith(('"""', "'''")):
            quote = lines[insert_idx].lstrip()[:3]
            for i in range(insert_idx+1, len(lines)):
                if lines[i].strip().endswith(quote):
                    insert_idx = i + 1
                    break

        else:
            break

    return insert_idx


def update_all(file_path: Path, names, tree):
    if not names:
        return

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove existing __all__ safely
    lines = remove_existing_all(lines, tree)

    # Generate new __all__ (multiline, readable)
    all_block = ["__all__ = (\n"]
    for name in names:
        all_block.append(f'    "{name}",\n')
    all_block.append(")\n\n")

    insert_idx = find_insert_position(lines)

    new_lines = lines[:insert_idx] + all_block + lines[insert_idx:]

    with file_path.open("w", encoding="utf-8") as f:
        f.writelines(new_lines)


def process_project(root: Path):
    for path in root.rglob("*.py"):
        names, tree = get_public_api(path)
        update_all(path, names, tree)
        print(f"Updated {path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    process_project(root)
