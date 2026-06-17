"""
Flake8 plugin to enforce code maintainability standards using Radon.
"""

try:
    from radon.metrics import mi_rank, mi_visit
except ModuleNotFoundError as exc:
    _exc = exc

    def mi_visit(*args, **kwds):
        """Rank the score with a letter."""
        raise _exc


class MaintainabilityChecker:
    """Flake8 plugin class to check the Maintainability Index of Python files"""

    name = "flake8-maintain"
    version = "0.1.0"

    DEFAULT_THRESHOLD = 19

    @classmethod
    def add_options(cls, parser):
        """Registers the custom command-line option for the maintainability checks"""
        parser.add_option(
            "--min-maintain-index",
            type=float,
            parse_from_config=True,
            default=cls.DEFAULT_THRESHOLD,
            help=(f"Minimum Maintainability Index allowed (default: {cls.DEFAULT_THRESHOLD})"),
        )

    @classmethod
    def parse_options(cls, options):
        """Parses and stores the maintainability threshold from options"""
        cls.threshold = options.min_maintain_index

    def __init__(self, tree, lines):
        """Initializes the checker with the tree mode"""
        self.lines = lines

    def run(self):
        """Calculates the Maintainability Index and yields a violation if below threshold."""
        try:
            score = mi_visit("".join(self.lines), multi=True)
        except Exception:
            return

        if score <= self.threshold:
            yield (
                1,  # line
                0,  # column
                f"MI100 Maintainability index {mi_rank(score)!r} ({score:.2f})"
                f" is below threshold ({self.threshold})",
                type(self),
            )
