"""Flake8 plugin to enforce source code metrics standards using Radon."""

import ast
from argparse import Namespace
from collections.abc import Generator
from typing import Any

from flake8.options.manager import OptionManager

try:
    from radon.metrics import mi_rank, mi_visit
except ModuleNotFoundError as exc:
    _exc = exc

    def mi_visit(*args: Any, **kwds: Any) -> float:
        """Visit the code and compute metric."""
        raise _exc

    def mi_rank(*args: Any, **kwds: Any) -> str:
        """Rank the score with a letter."""
        raise _exc


class MetricsChecker:
    """Flake8 plugin class to check source code metrics of Python files"""

    name = "flake8-metrics"
    version = "1.0.0"

    @classmethod
    def add_options(cls, parser: OptionManager) -> None:
        """Registers the custom command-line option for the metrics checks"""
        parser.add_option(
            "--min-maintain-index",
            type=float,
            parse_from_config=True,
            default=19,
            help=("Minimum Maintainability Index allowed"),
        )

    @classmethod
    def parse_options(cls, options: Namespace) -> None:
        """Parses and stores the metrics threshold from options"""
        cls.threshold = options.min_maintain_index

    def __init__(self, tree: ast.AST, lines: list[str]) -> None:
        """Initializes the checker with the tree mode"""
        self.lines = lines

    def run(self) -> Generator[tuple[int, int, str, type]]:
        """Calculates the metrics and yields a violation if below threshold."""
        try:
            score = mi_visit("".join(self.lines), multi=True)
        except Exception:
            return

        if score <= self.threshold:
            yield (
                1,  # line
                0,  # column
                f"M100 Maintainability index {mi_rank(score)!r} ({score:.2f})"
                f" is below threshold ({self.threshold})",
                type(self),
            )
