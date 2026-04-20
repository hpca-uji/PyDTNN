"""PyDTNN environment initialization"""

import logging
logger = logging.getLogger(__name__)

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
package_name = __name__

# OPTIONAL IMPORTS
try:
    from pydtnn.libs.mpi import MPI
except Exception:
    MPI = None

try:
    import cupy  # type: ignore
    logger.debug("Cupy available")
except Exception as e:
    logger.debug(f"Cupy not available\n{e}")
    cupy = None
    gpu_errors.append(e)

try:
    import pycuda  # type: ignore
    logger.debug("PyCuda available")
    import pycuda.driver as drv  # type: ignore
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
    import pycuda.gpuarray as gpuarray  # type: ignore
    has_drv = True

try:
    from pydtnn.backends.pycuda.utils import tensor_array  # type: ignore
    logger.debug(f"tensor_array available")
except Exception as e:
    logger.debug(f"tensor_array not available\n{e}")
    tensor_array = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import nccl as nccl  # type: ignore
    logger.debug(f"nccl available")
except Exception as e:
    logger.debug(f"nccl not available\n{e}")
    nccl = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import cudnn as cudnn  # type: ignore
    logger.debug(f"cudnn available")
except Exception as e:
    logger.debug(f"cudnn not available\n{e}")
    cudnn = None
    gpu_errors.append(e)

try:
    from pydtnn.libs import cublas  # type: ignore
    logger.debug(f"cublas available")
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

# INIT CUPY
if cupy is not None and drv is not None:
    rank = MPI.COMM_WORLD.rank if MPI else 0
    cupy.cuda.runtime.setDevice(rank % cupy.cuda.runtime.getDeviceCount())
    stream: cupy.cuda.Stream = cupy.cuda.get_current_stream()
    stream_handle = stream.ptr
else:
    pass  # Defaults handled later

# INIT PYCUDA
if drv is not None:
    drv.init()
    rank = MPI.COMM_WORLD.rank if MPI else 0
    device = drv.Device(rank % drv.Device.count())
    context = device.make_context()
    stream: drv.Stream = drv.Stream()  # type: ignore
    stream_handle = stream.handle
    atexit.register(lambda: context.pop())  # type: ignore
else:
    context = None  # type: ignore
    # Defaults handled later

# DEFAULT CUDA
if cupy is None and drv is None:
    device = None  # type: ignore
    context = None  # type: ignore
    stream = None  # type: ignore
    stream_handle = None  # type: ignore
# ---

# INIT CUDNN
if cudnn is not None and drv is not None:
    # NOTE: CUDNN initalization must be done after "drv.init()"
    cudnn_handle: Cudnn_Handle_Type = cudnn.cudnnCreate()  # type: ignore
    atexit.register(lambda: cudnn.cudnnDestroy(cudnn_handle))  # type: ignore
else:
    cudnn_handle: Cudnn_Handle_Type = None  # type: ignore
# ---

# INIT CUBLAS
if cublas is not None and device is not None:
    cublas_handle: Cublas_Handle_Type = cublas.cublasCreate()  # type: ignore
    atexit.register(lambda: cublas.cublasDestroy(cublas_handle))  # type: ignore
else:
    cublas_handle: Cublas_Handle_Type = None  # type: ignore
# ---

# SYNC CUDNN+CUDA
if cudnn is not None and stream is not None:
    cudnn.cudnnSetStream(cudnn_handle, stream_handle)
# ---

# SYNC CUBLAS+CUDA
if cublas is not None and stream is not None:
    cublas.cublasSetStream(cublas_handle, stream_handle)  # type: ignore
# ---
