"""MPI server-client test"""

from argparse import ArgumentParser, Namespace


__all__ = ()


# Argument pasrser
parser = ArgumentParser(prog="test_libs_mpi", description="MPI server-client test")


def main(config: Namespace):
    """Application entrypoint"""
    from pydtnn.libs.mpi import client as MPI

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    print(f"R{rank}: size {size}")

    comm.barrier()
    print(f"R{rank}: barrier")

    root = 0
    ref = root
    res = comm.bcast(rank, root=root)
    print(f"R{rank}: bcast {res}/{ref}")
    assert res == ref, f"bcast error; got {res}, expect {ref}"

    ref = list(range(size))
    res = comm.allgather(rank)
    print(f"R{rank}: allgather {res}/{ref}")
    assert res == ref, f"allgather error; got {res}, expect {ref}"

    ref = sum(range(size))
    res = comm.allreduce(rank)
    print(f"R{rank}: allreduce {res}/{ref}")
    assert res == ref, f"allreduce error; got {res}, expect {ref}"

    root = 0
    ref = rank
    res = comm.scatter(range(size), root=root)
    print(f"R{rank}: scatter {res}/{ref}")
    assert res == ref, f"scatter error; got {res}, expect {ref}"

    ref = [rank] * size
    res = comm.alltoall(range(size))
    print(f"R{rank}: alltoall {res}/{ref}")
    assert res == ref, f"alltoall error; got {res}, expect {ref}"

    root = 0
    ref = list(range(size)) if rank == root else None
    res = comm.gather(rank, root=root)
    print(f"R{rank}: gather {res}/{ref}")
    assert res == ref, f"gather error; got {res}, expect {ref}"

    root = 0
    ref = sum(range(size)) if rank == root else None
    res = comm.reduce(rank, root=root)
    print(f"R{rank}: reduce {res}/{ref}")
    assert res == ref, f"reduce error; got {res}, expect {ref}"

    ref = rank
    prev = (rank - 1) % size
    next = (rank + 1) % size
    if size > 1:
        comm.send(next, dest=next)
        res = comm.recv(source=prev)
    else:
        res = ref
    print(f"R{rank}: send/recv {res}/{ref}")
    assert res == ref, f"send/recv error; got {res}, expect {ref}"

    ref = rank
    prev = (rank - 1) % size
    next = (rank + 1) % size
    if size > 1:
        res = comm.sendrecv(next, dest=next, source=prev)
    else:
        res = ref
    print(f"R{rank}: sendrecv {res}/{ref}")
    assert res == ref, f"sendrecv error; got {res}, expect {ref}"

    MPI.Finalize()
    print(f"R{rank}: finalize")


if __name__ == "__main__":
    main(parser.parse_args())
