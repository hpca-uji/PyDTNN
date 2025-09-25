"""MPI server-client IOPS test"""

import time
import enum
from argparse import ArgumentParser, Namespace

import numpy

from pydtnn import utils


__all__ = ()


class Mode(enum.StrEnum):
    """Test mode"""
    SEQUENTIAL = enum.auto()
    RANDOM = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_comms_iops", description="MPI server-client IOPS test")
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--delay", type=float, default=0.0)
parser.add_argument("--size", type=int, default=1_000)
parser.add_argument("--reps", type=int, default=100_000)


def print_stats(config: Namespace, time: float) -> None:
    """Print statistics"""
    ops = config.reps * 2
    size = config.size * ops
    print(f"Time:       {time:.1f}s")
    print(f"Data:       {utils.convert_size(config.reps)} x {utils.convert_size(config.size)}B")
    print(f"Transfer:   {utils.convert_size(size)}B @ {utils.convert_size(size * 8 / time):>5}bps")
    print(f"Operations: {utils.convert_size(ops)} @ {utils.convert_size(ops / time)}IOPS")


def main(config: Namespace):
    """Application entrypoint"""
    message = numpy.arange(config.size, dtype=numpy.uint8)

    from pympi import MPI

    comm = MPI.COMM_WORLD
    size = comm.size

    messages = [ary.copy() for ary in numpy.split(message, size)]

    comm.barrier()
    start_time = time.time()

    match config.mode:
        case Mode.SEQUENTIAL:
            for _ in range(config.reps):
                comm.alltoall(messages)

        case Mode.RANDOM:
            reqs = [comm.ialltoall(messages) for _ in range(config.reps)]
            while reqs:
                done = set(MPI.Request.Waitsome(reqs))
                reqs = [req for i, req in enumerate(reqs) if i not in done]

    end_time = time.time()
    del message
    MPI.Finalize()

    print_stats(config=config, time=end_time - start_time)


if __name__ == "__main__":
    main(parser.parse_args())
