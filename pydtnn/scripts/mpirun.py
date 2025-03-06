#!/usr/bin/env python3
"""Basic local mpirun"""

import os
import typing
import subprocess
from argparse import ArgumentParser, Namespace


__all__ = (
    "main",
)


arg_parser = ArgumentParser(
    prog="mpirun",
    description="basic local mpirun"
)

arg_parser.add_argument("-np", dest="size", type=int)


def main(*args: str) -> None:
    """Application entrypoint"""
    config, program = arg_parser.parse_known_args(args)
    config = typing.cast(Namespace, config)

    for rank in range(config.size):
        environment = os.environ.copy()
        environment.update({"PMI_RANK": str(rank), "PMI_SIZE": str(config.size)})
        subprocess.Popen(args=program, env=environment)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
