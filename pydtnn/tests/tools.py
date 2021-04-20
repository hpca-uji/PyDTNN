"""
Common tools used by the unit tests.
"""

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def print_with_header(header, to_be_printed=None):
    _console = Console(force_terminal=True)
    panel = Panel.fit(Text(header, justify="right"), style="bold blue")
    _console.print()
    _console.print(panel)
    if to_be_printed is not None:
        print(to_be_printed)


if __name__ == "__main__":
    import time

    print_with_header("Testing the spinner (3 seconds)")
    console = Console(force_terminal=True)
    with console.status("", spinner="dots10"):
        time.sleep(3)
    print("Done!")
