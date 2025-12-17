"""PyDTNN environment initialization"""

import os
import atexit
import platform
import subprocess
from types import ModuleType
from collections import Counter

type MPI_MODULE = ModuleType
type Cudnn_Handle_Type = int
type Cublas_Handle_Type = int

gpu_errors = []

# OPTIONAL IMPORTS
try:
    from pydtnn.libs.libmpi import MPI
except Exception:
    MPI = None

try:
    import pycuda.driver as drv  # type: ignore
except Exception as e:
    drv = None
    gpu_errors.append(e)
    has_drv = False
else:
    has_drv = True

try:
    import pycuda.gpuarray as gpuarray  # type: ignore
except Exception as e:
    gpuarray = None
    gpu_errors.append(e)

try:
    from pydtnn.backends.gpu.utils import tensor_gpu  # type: ignore
except Exception as e:
    tensor_gpu = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import libnccl as nccl  # type: ignore
except Exception as e:
    nccl = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import libcudnn as cudnn  # type: ignore
except Exception as e:
    cudnn = None
    gpu_errors.append(e)

try:
    from skcuda import cublas  # type: ignore
except Exception as e:
    cublas = None
    gpu_errors.append(e)


# INIT MPI
if MPI is not None:
    rank = MPI.COMM_WORLD.rank
    nprocs = MPI.COMM_WORLD.size
    hostname = platform.node()
    ranks_per_node = dict(Counter(MPI.COMM_WORLD.allgather(hostname)))
else:
    rank = 0
    nprocs = 1
    hostname = "localhost"
    ranks_per_node = {hostname: nprocs}
# ---

# INIT GPU
try:
    num_gpus = subprocess.check_output(["nvidia-smi", "-L"]).count(b'UUID')
except (FileNotFoundError, subprocess.CalledProcessError):
    num_gpus = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus) if num_gpus else ""
supported_gpu = bool(num_gpus)
# ---

# INIT NCCL
if nccl is not None and num_gpus > 0:
    nccl_id = nccl.ncclGetUniqueId()
    if MPI:
        nccl_id = MPI.COMM_WORLD.bcast(nccl_id)
    nccl_comm = nccl.ncclCommInitRank(nprocs, nccl_id, rank)
    atexit.register(lambda: nccl.ncclCommDestroy(nccl_comm))  # type: ignore
else:
    nccl_comm = None  # type: ignore
# ---

# INIT PYCUDA
if drv is not None:  # equivalent: if has_drv:
    drv.init()
    rank = MPI.COMM_WORLD.rank if MPI else 0
    device = drv.Device(rank % drv.Device.count())
    context = device.make_context()
    stream: drv.Stream = drv.Stream()  # type: ignore
    atexit.register(lambda: context.detach())  # type: ignore
else:
    device = None  # type: ignore
    context = None  # type: ignore
    stream = None  # type: ignore
# ---

# INIT CUDNN
if cudnn is not None and has_drv:
    # NOTE: CUDNN initalization must be done after "drv.init()"
    cudnn_handle: Cudnn_Handle_Type = cudnn.cudnnCreate()  # type: ignore
    atexit.register(lambda: cudnn.cudnnDestroy(cudnn_handle))  # type: ignore
else:
    cudnn_handle: Cudnn_Handle_Type = None  # type: ignore
# ---

# INIT CUBLAS
if cublas is not None:
    cublas_handle: Cublas_Handle_Type = cublas.cublasCreate()  # type: ignore
    atexit.register(lambda: cublas.cublasDestroy(cublas_handle))  # type: ignore
else:
    cublas_handle: Cublas_Handle_Type = None  # type: ignore
# ---

# SYNC CUDNN+PYCUDA
if cudnn is not None and stream is not None:
    cudnn.cudnnSetStream(cudnn_handle, stream.handle)
# ---

# SYNC CUBLAS+PYCUDA
if cublas is not None and stream is not None:
    cublas.cublasSetStream(cublas_handle, stream.handle)  # type: ignore
# ---
