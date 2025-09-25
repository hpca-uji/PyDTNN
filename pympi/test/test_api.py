"""MPI server-client API test"""

from argparse import ArgumentParser, Namespace


__all__ = ()


# Argument pasrser
parser = ArgumentParser(prog="test_libs_mpi", description="MPI server-client API test")
parser.add_argument("--rank-offset", type=int, default=45)


def main(config: Namespace):
    """Application entrypoint"""
    from pympi import MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    prefix = " " * config.rank_offset * rank + f"R{rank}:"
    print(prefix, f"size {size}")

    comm.barrier()
    print(prefix, "barrier")

    root = 0
    ref = root
    print(prefix, f"bcast /{ref}")
    res = comm.bcast(rank, root=root)
    print(prefix, f"bcast {res}/{ref}")
    assert res == ref, f"bcast error; got {res}, expect {ref}"

    ref = list(range(size))
    res = comm.allgather(rank)
    print(prefix, f"allgather {res}/{ref}")
    assert res == ref, f"allgather error; got {res}, expect {ref}"

    ref = sum(range(size))
    res = comm.allreduce(rank)
    print(prefix, f"allreduce {res}/{ref}")
    assert res == ref, f"allreduce error; got {res}, expect {ref}"

    root = 0
    ref = rank
    res = comm.scatter(list(range(size)), root=root)
    print(prefix, f"scatter {res}/{ref}")
    assert res == ref, f"scatter error; got {res}, expect {ref}"

    ref = [rank] * size
    res = comm.alltoall(list(range(size)))
    print(prefix, f"alltoall {res}/{ref}")
    assert res == ref, f"alltoall error; got {res}, expect {ref}"

    root = 0
    ref = list(range(size)) if rank == root else None
    res = comm.gather(rank, root=root)
    print(prefix, f"gather {res}/{ref}")
    assert res == ref, f"gather error; got {res}, expect {ref}"

    root = 0
    ref = sum(range(size)) if rank == root else None
    res = comm.reduce(rank, root=root)
    print(prefix, f"reduce {res}/{ref}")
    assert res == ref, f"reduce error; got {res}, expect {ref}"

    ref = rank
    prev = (rank - 1) % size
    next = (rank + 1) % size
    if size > 1:
        comm.send(next, dest=next)
        res = comm.recv(source=prev)
        pass
    else:
        res = ref
    print(prefix, f"send/recv {res}/{ref}")
    assert res == ref, f"send/recv error; got {res}, expect {ref}"

    ref = rank
    prev = (rank - 1) % size
    next = (rank + 1) % size
    if size > 1:
        res = comm.sendrecv(next, dest=next, source=prev)
    else:
        res = ref
    print(prefix, f"sendrecv {res}/{ref}")
    assert res == ref, f"sendrecv error; got {res}, expect {ref}"

    MPI.Finalize()
    print(prefix, "finalize")


if __name__ == "__main__":
    main(parser.parse_args())
