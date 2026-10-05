"""Test MPI CUDA-aware support"""

import numpy as np
from mpi4py import MPI
import pycuda.autoinit
from pycuda import gpuarray

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

x = gpuarray.to_gpu(np.arange(10) * (10**rank))
print(rank, x)
comm.Iallreduce(MPI.IN_PLACE, x).wait()
print(rank, x)
