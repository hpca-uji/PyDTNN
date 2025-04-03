"""MPI server-client test"""

import enum
from argparse import ArgumentParser, Namespace


__all__ = ()


class Mode(enum.StrEnum):
    """Test modes"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_libs_mpi", description="MPI server-client test")


def main(config: Namespace):
    """Application entrypoint"""
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

    MPI.Finalize()


if __name__ == "__main__":
    main(parser.parse_args())
