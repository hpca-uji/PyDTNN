"""MPI server-client exception test"""

from argparse import ArgumentParser, Namespace


__all__ = ()


# Argument pasrser
parser = ArgumentParser(prog="test_libs_mpi_exc", description="MPI server-client exception test")
parser.add_argument("--rank-offset", type=int, default=45)


def main(config: Namespace):
    """Application entrypoint"""
    from pympi import MPI
    from pympi.proto import RemoteException

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    prefix = " " * config.rank_offset * rank + f"R{rank}:"
    print(prefix, f"size {size}")

    ref = 0
    res = comm.bcast(rank, root=ref)
    print(prefix, f"bcast {res}/{ref}")
    assert res == ref, f"bcast error; got {res}, expect {ref}"

    ref = RemoteException if size > 1 else type(None)
    try:
        res = comm.allreduce(None)
    except RemoteException as exc:
        res = exc
    print(prefix, f"error handeling {type(res)}/{ref}")
    assert isinstance(res, ref), f"error handeling error; got {res}, expect {ref}"

    ref = None
    try:
        comm.barrier()
    except Exception as exc:
        res = exc
    else:
        res = None
    print(prefix, f"error recovery {res}/{ref}")
    assert res == ref, f"error recovery error; got {res}, expect {ref}"

    MPI.Finalize()
    print(prefix, "finalize")


if __name__ == "__main__":
    main(parser.parse_args())
