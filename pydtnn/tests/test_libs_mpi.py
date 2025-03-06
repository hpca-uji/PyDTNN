"""MPI server-client test"""

from argparse import ArgumentParser, Namespace


__all__ = ()


# Argument pasrser
parser = ArgumentParser(prog="test_libs_mpi", description="MPI server-client test")


def main(config: Namespace):
    """Application entrypoint"""
    from pydtnn.libs.mpi import client as MPI
    from pydtnn.libs.mpi.comm import RemoteException

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

    ref = RemoteException
    try:
        res = comm.allreduce(None, MPI.SUM)
    except RemoteException as exc:
        res = exc
    print(f"R{rank}: error handeling {type(res)}/{ref}")
    assert isinstance(res, ref), f"error handeling error; got {res}, expect {ref}"

    ref = None
    try:
        comm.barrier()
    except Exception as exc:
        res = exc
    else:
        res = None
    print(f"R{rank}: error recovery {res}/{ref}")
    assert res == ref, f"error recovery error; got {res}, expect {ref}"

    MPI.Finalize()
    print(f"R{rank}: finalize")


if __name__ == "__main__":
    main(parser.parse_args())
