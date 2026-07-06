"""Flake8 plugin for Pyright."""

import ast
import enum
import itertools
import json
import re
from argparse import Namespace
from collections.abc import Generator, Iterable
from pathlib import Path

import pyright
from flake8.discover_files import expand_paths

_re_pascal = re.compile("[a-z][A-Z]")


def pascal_snake(pascal: str) -> str:
    """Converts a pascal case string to snake case"""
    return _re_pascal.sub(lambda b: "_".join(b[0]), pascal).lower()


def run_pyright(filenames: Iterable[str] = []) -> dict[str, list[dict]]:
    """Run pyright and group results by filename"""
    output = json.loads(
        pyright.run(
            "--outputjson", "-",
            input="\n".join(filenames),
            text=True, capture_output=True
        ).stdout
    )

    def key(diagnostic):
        return diagnostic["file"]

    diagnostics: list[dict] = output["generalDiagnostics"]
    diagnostics.sort(key=key)
    return {
        file: list(diagnostics)
        for file, diagnostics in itertools.groupby(diagnostics, key=key)
    }


class DiagnosticRule(enum.IntEnum):
    """Pyright diagnostic rules"""

    REPORT_GENERAL_TYPE_ISSUES = enum.auto()
    REPORT_PROPERTY_TYPE_MISMATCH = enum.auto()
    REPORT_FUNCTION_MEMBER_ACCESS = enum.auto()
    REPORT_MISSING_IMPORTS = enum.auto()
    REPORT_MISSING_MODULE_SOURCE = enum.auto()
    REPORT_INVALID_TYPE_FORM = enum.auto()
    REPORT_MISSING_TYPE_STUBS = enum.auto()
    REPORT_IMPORT_CYCLES = enum.auto()
    REPORT_UNUSED_IMPORT = enum.auto()
    REPORT_UNUSED_CLASS = enum.auto()
    REPORT_UNUSED_FUNCTION = enum.auto()
    REPORT_UNUSED_VARIABLE = enum.auto()
    REPORT_DUPLICATE_IMPORT = enum.auto()
    REPORT_WILDCARD_IMPORT_FROM_LIBRARY = enum.auto()
    REPORT_ABSTRACT_USAGE = enum.auto()
    REPORT_ARGUMENT_TYPE = enum.auto()
    REPORT_ASSERT_TYPE_FAILURE = enum.auto()
    REPORT_ASSIGNMENT_TYPE = enum.auto()
    REPORT_ATTRIBUTE_ACCESS_ISSUE = enum.auto()
    REPORT_CALL_ISSUE = enum.auto()
    REPORT_INCONSISTENT_OVERLOAD = enum.auto()
    REPORT_INDEX_ISSUE = enum.auto()
    REPORT_INVALID_TYPE_ARGUMENTS = enum.auto()
    REPORT_NO_OVERLOAD_IMPLEMENTATION = enum.auto()
    REPORT_OPERATOR_ISSUE = enum.auto()
    REPORT_OPTIONAL_SUBSCRIPT = enum.auto()
    REPORT_OPTIONAL_MEMBER_ACCESS = enum.auto()
    REPORT_OPTIONAL_CALL = enum.auto()
    REPORT_OPTIONAL_ITERABLE = enum.auto()
    REPORT_OPTIONAL_CONTEXT_MANAGER = enum.auto()
    REPORT_OPTIONAL_OPERAND = enum.auto()
    REPORT_REDECLARATION = enum.auto()
    REPORT_RETURN_TYPE = enum.auto()
    REPORT_TYPED_DICT_NOT_REQUIRED_ACCESS = enum.auto()
    REPORT_UNTYPED_FUNCTION_DECORATOR = enum.auto()
    REPORT_UNTYPED_CLASS_DECORATOR = enum.auto()
    REPORT_UNTYPED_BASE_CLASS = enum.auto()
    REPORT_UNTYPED_NAMED_TUPLE = enum.auto()
    REPORT_PRIVATE_USAGE = enum.auto()
    REPORT_TYPE_COMMENT_USAGE = enum.auto()
    REPORT_PRIVATE_IMPORT_USAGE = enum.auto()
    REPORT_CONSTANT_REDEFINITION = enum.auto()
    REPORT_DEPRECATED = enum.auto()
    REPORT_INCOMPATIBLE_METHOD_OVERRIDE = enum.auto()
    REPORT_INCOMPATIBLE_VARIABLE_OVERRIDE = enum.auto()
    REPORT_INCONSISTENT_CONSTRUCTOR = enum.auto()
    REPORT_OVERLAPPING_OVERLOAD = enum.auto()
    REPORT_POSSIBLY_UNBOUND_VARIABLE = enum.auto()
    REPORT_MISSING_SUPER_CALL = enum.auto()
    REPORT_UNINITIALIZED_INSTANCE_VARIABLE = enum.auto()
    REPORT_INVALID_STRING_ESCAPE_SEQUENCE = enum.auto()
    REPORT_UNKNOWN_PARAMETER_TYPE = enum.auto()
    REPORT_UNKNOWN_ARGUMENT_TYPE = enum.auto()
    REPORT_UNKNOWN_LAMBDA_TYPE = enum.auto()
    REPORT_UNKNOWN_VARIABLE_TYPE = enum.auto()
    REPORT_UNKNOWN_MEMBER_TYPE = enum.auto()
    REPORT_MISSING_PARAMETER_TYPE = enum.auto()
    REPORT_MISSING_TYPE_ARGUMENT = enum.auto()
    REPORT_INVALID_TYPE_VAR_USE = enum.auto()
    REPORT_CALL_IN_DEFAULT_INITIALIZER = enum.auto()
    REPORT_UNNECESSARY_IS_INSTANCE = enum.auto()
    REPORT_UNNECESSARY_CAST = enum.auto()
    REPORT_UNNECESSARY_COMPARISON = enum.auto()
    REPORT_UNNECESSARY_CONTAINS = enum.auto()
    REPORT_ASSERT_ALWAYS_TRUE = enum.auto()
    REPORT_SELF_CLS_PARAMETER_NAME = enum.auto()
    REPORT_IMPLICIT_STRING_CONCATENATION = enum.auto()
    REPORT_UNDEFINED_VARIABLE = enum.auto()
    REPORT_UNBOUND_VARIABLE = enum.auto()
    REPORT_UNHASHABLE = enum.auto()
    REPORT_INVALID_STUB_STATEMENT = enum.auto()
    REPORT_INCOMPLETE_STUB = enum.auto()
    REPORT_UNSUPPORTED_DUNDER_ALL = enum.auto()
    REPORT_UNUSED_CALL_RESULT = enum.auto()
    REPORT_UNUSED_COROUTINE = enum.auto()
    REPORT_UNUSED_EXCEPT = enum.auto()
    REPORT_UNUSED_EXPRESSION = enum.auto()
    REPORT_UNNECESSARY_TYPE_IGNORE_COMMENT = enum.auto()
    REPORT_MATCH_NOT_EXHAUSTIVE = enum.auto()
    REPORT_UNREACHABLE = enum.auto()
    REPORT_IMPLICIT_OVERRIDE = enum.auto()


class PyrightChecker:
    """Flake8 plugin class to check the Maintainability Index of Python files"""

    name = "flake8-pyright"
    version = "1.0.0"

    @classmethod
    def parse_options(cls, options: Namespace) -> None:
        """Parses and stores the maintainability threshold from options"""
        cls._diagnostics = run_pyright(
            expand_paths(
                paths=options.filenames,
                stdin_display_name=options.stdin_display_name,
                filename_patterns=options.filename,
                exclude=(*options.exclude, *options.extend_exclude),
            )
        )

    def __init__(self, tree: ast.AST, filename: str) -> None:
        """Initializes the checker with the tree mode"""
        self.filename = str(Path(filename).resolve())

    def run(self) -> Generator[tuple[int, int, str, type]]:
        """Calculates the Maintainability Index and yields a violation if below threshold."""
        for diagnostic in self._diagnostics.get(self.filename, []):
            rule_name = pascal_snake(diagnostic["rule"]).upper()
            try:
                rule = DiagnosticRule[rule_name]
            except KeyError:
                print(rule_name)
                rule = DiagnosticRule.REPORT_GENERAL_TYPE_ISSUES

            yield (
                diagnostic["range"]["start"]["line"],
                diagnostic["range"]["start"]["character"],
                f"T{rule:03} {diagnostic['message'].splitlines()[0]}",
                type(self),
            )
