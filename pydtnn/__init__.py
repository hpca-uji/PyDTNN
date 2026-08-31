"""PyDTNN environment initialization"""

import atexit
import logging
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime
from types import ModuleType

__all__ = (
    "MPI_MODULE",
    "Cudnn_Handle_Type",
    "Cublas_Handle_Type",
    "gpu_errors",
    "package_name",
    "timestamp",
    "MPI",
    "cupy",
    "pycuda",
    "drv",
    "gpuarray",
    "stream",
    "has_drv",
    "tensor_array",
    "nccl",
    "cudnn",
    "cublas",
    "rank",
    "nprocs",
    "hostname",
    "ranks_per_node",
    "num_gpus",
    "supported_gpu",
    "nccl_comm",
    "context",
    "stream_handle",
    "cudnn_handle",
    "cublas_handle",
)

logger = logging.getLogger(__name__)


type MPI_MODULE = ModuleType
type Cudnn_Handle_Type = int
type Cublas_Handle_Type = int

gpu_errors = []
package_name = __name__
timestamp = (
    datetime.now()
    .isoformat(timespec="seconds")
    .replace(" ", "-")
    .replace(":", "-")
    .replace(".", "-")
)

# OPTIONAL IMPORTS
try:
    from pydtnn.libs.mpi import MPI
except Exception:
    MPI = None

try:
    import cupy

    logger.debug("Cupy available")
except Exception as e:
    logger.debug(f"Cupy not available\n{e}")
    cupy = None
    gpu_errors.append(e)

try:
    import pycuda

    logger.debug("PyCuda available")

    import pycuda.driver as drv

    logger.debug("drv available")
except Exception as e:
    logger.debug(f"PyCuda or drv not available\n{e}")
    gpu_errors.append(e)
    pycuda = None
    drv = None
    gpuarray = None
    stream = None
    has_drv = False
else:
    import pycuda.gpuarray as gpuarray

    has_drv = True

try:
    from pydtnn.backends.pycuda.utils import tensor_array

    logger.debug("tensor_array available")
except Exception as e:
    logger.debug(f"tensor_array not available\n{e}")
    tensor_array = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import nccl as nccl

    logger.debug("nccl available")
except Exception as e:
    logger.debug(f"nccl not available\n{e}")
    nccl = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import cudnn as cudnn

    logger.debug("cudnn available")
except Exception as e:
    logger.debug(f"cudnn not available\n{e}")
    cudnn = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import cublas

    logger.debug("cublas available")
except Exception as e:
    logger.debug(f"cublas available\n{e}")
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

# INIT GPU
try:
    num_gpus = subprocess.check_output(["nvidia-smi", "-L"]).count(b"UUID")
except (FileNotFoundError, subprocess.CalledProcessError):
    num_gpus = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % num_gpus) if num_gpus else ""
supported_gpu = bool(num_gpus)

# INIT NCCL
if nccl is not None and num_gpus > 0:
    nccl_id = nccl.ncclGetUniqueId()
    if MPI:
        nccl_id = MPI.COMM_WORLD.bcast(nccl_id)
    nccl_comm = nccl.ncclCommInitRank(nprocs, nccl_id, rank)
    nccl_destroy = nccl.ncclCommDestroy

    def _destory() -> None:
        assert nccl and nccl_comm
        nccl.ncclCommDestroy(nccl_comm)

    atexit.register(_destory)
else:
    nccl_comm = None

# DEFAULT CUDA
device = None
context = None
stream = None
stream_handle = None

# INIT CUPY
if cupy is not None and drv is not None:
    from cupy.cuda import Stream as CupyStream

    cupy.cuda.runtime.setDevice(rank % cupy.cuda.runtime.getDeviceCount())
    stream: CupyStream = cupy.cuda.get_current_stream()
    stream_handle = stream.ptr

# INIT PYCUDA
if drv is not None:
    drv.init()
    from pycuda.driver import Stream as PycudaStream

    device = drv.Device(rank % num_gpus)
    context = device.make_context()
    stream: PycudaStream = drv.Stream()
    stream_handle = stream.handle

    def _destroy() -> None:
        assert context
        context.pop()

    atexit.register(_destroy)

# INIT CUDNN
if cudnn is not None and drv is not None:
    # NOTE: CUDNN initalization must be done after "drv.init()"
    cudnn_handle: Cudnn_Handle_Type = cudnn.cudnnCreate()

    def _destroy() -> None:
        assert cudnn and cudnn_handle
        cudnn.cudnnDestroy(cudnn_handle)

    atexit.register(_destroy)
else:
    cudnn_handle = None  # pyright: ignore[reportAssignmentType]


# INIT CUBLAS
if cublas is not None and device is not None:
    cublas_handle: Cublas_Handle_Type = cublas.cublasCreate()

    def _destroy() -> None:
        assert cublas, cublas_handle
        cublas.cublasDestroy(cublas_handle)

    atexit.register(_destroy)
else:
    cublas_handle: Cublas_Handle_Type = None  # pyright: ignore[reportAssignmentType]

# SYNC CUDNN+CUDA
if cudnn is not None and stream_handle is not None:
    cudnn.cudnnSetStream(cudnn_handle, stream_handle)

# SYNC CUBLAS+CUDA
if cublas is not None and stream_handle is not None:
    cublas.cublasSetStream(cublas_handle, stream_handle)
