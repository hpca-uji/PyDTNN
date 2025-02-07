"""MPI server-client test"""

import sys
import enum
import time
from argparse import ArgumentParser, Namespace


__all__ = ()


class Mode(enum.StrEnum):
    """Test modes"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_libs_mpi", description="MPI server-client test")
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--delay", type=float, default=3.0)


def server(config: Namespace):
    """Server mode"""
    from pydtnn.libs.mpi.server import Server

    with Server() as server:
        server.serve_forever()


def client(config):
    """Client mode"""
    time.sleep(config.delay)

    from pydtnn.libs.mpi import client as MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    print(f"R{rank}: size {size}")

    ref = 0
    res = comm.bcast(rank, root=ref)
    print(f"R{rank}: bcast {res}/{ref}")
    assert res == ref, f"bcast error; got {res}, expect {ref}"

    ref = list(range(size))
    res = comm.allgather(rank)
    print(f"R{rank}: allgather {res}/{ref}")
    assert res == ref, f"allgather error; got {res}, expect {ref}"

    ref = sum(rank for rank in range(size))
    res = comm.allreduce(rank, MPI.SUM)
    print(f"R{rank}: allreduce {res}/{ref}")
    assert res == ref, f"allreduce error; got {res}, expect {ref}"

    comm.Disconnect()


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
