#!/usr/bin/env python3
"""Basic local mpirun"""

import os
import subprocess
from collections import abc
from argparse import ArgumentParser, Namespace


__all__ = (
    "main",
)


arg_parser = ArgumentParser(
    prog="mpirun",
    description="basic local mpirun"
)

arg_parser.add_argument("-np", dest="size", type=int)


def main(config: Namespace, args: abc.Sequence[str]) -> None:
    """Application entrypoint"""

    for rank in range(config.size):
        environ = os.environ.copy()
        environ.update({"PMI_RANK": str(rank), "PMI_SIZE": str(config.size)})
        subprocess.Popen(args=args, env=environ)


if __name__ == "__main__":
    import sys
    config, args = arg_parser.parse_known_args(*sys.argv[1:])
    main(config, args)  # type: ignore
