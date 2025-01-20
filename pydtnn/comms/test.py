from pydtnn.comms import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Logic check
result = comm.allgather(rank)
expect = list(range(size))
assert result == expect, f"R{rank}: {result!r} (result) != {expect!r} (expected)"

# Sync check
print(f"R{rank}: barrier (before)")
comm.Barrier()
print(f"R{rank}: barrier (after)")

MPI.Finalize()
