"""PyDTNN environment initialization"""

import atexit
import itertools
import logging
import os
import platform
import uuid
from collections import Counter
from datetime import datetime
from importlib import metadata as libdata

__all__ = (
    "cublas_handle",
    "cudnn_handle",
    "cupy_stream",
    "cupy_device",
    "errors",
    "host_uuid",
    "metadata",
    "nccl_comm",
    "mpi_comm",
    "nprocs",
    "package_name",
    "rank",
    "ranks_per_node",
    "state_path",
    "pycuda_stream",
    "pycuda_device",
    "timestamp",
)

logger = logging.getLogger(__name__)

errors = []
host_uuid = uuid.getnode()
package_name = __name__
state_path = os.environ.get(f"{package_name.upper()}_STATE", ".")
timestamp = (
    datetime.now()
    .isoformat(timespec="seconds")
    .replace(" ", "-")
    .replace(":", "-")
    .replace(".", "-")
)

cpu_affinity = tuple(range(os.process_cpu_count()))
try:
    _sched_getaffinity = os.sched_getaffinity
except AttributeError:
    pass  # unsupported
else:
    cpu_affinity = tuple(_sched_getaffinity(0))
try:
    from psutil import Process
except ModuleNotFoundError:
    pass  # unavailable
else:
    cpu_affinity = tuple(Process().cpu_affinity())

# OPTIONAL IMPORTS
try:
    from pydtnn.libs.mpi import MPI
except Exception as e:
    logger.debug("MPI not available", exc_info=e)
    errors.append(e)
    MPI = None
else:
    logger.debug("MPI available")

try:
    from pydtnn.libs import nccl
except Exception as e:
    logger.debug("nccl not available", exc_info=e)
    errors.append(e)
    nccl = None
else:
    logger.debug("nccl available")

try:
    import cupy.cuda as cupy_driver
except Exception as e:
    logger.debug("CuPy not available", exc_info=e)
    errors.append(e)
    cupy_driver = None
else:
    logger.debug("CuPy available")

try:
    import pycuda.driver as pycuda_driver
except Exception as e:
    logger.debug("PyCUDA not available", exc_info=e)
    errors.append(e)
    pycuda_driver = None
else:
    logger.debug("PyCUDA available")

try:
    from pydtnn.libs import cudnn
except Exception as e:
    logger.debug("cudnn not available", exc_info=e)
    errors.append(e)
    cudnn = None
else:
    logger.debug("cudnn available")

try:
    from pydtnn.libs import cublas
except Exception as e:
    logger.debug("cublas available", exc_info=e)
    errors.append(e)
    cublas = None
else:
    logger.debug("cublas available")

# INIT MPI
rank = 0
nprocs = 1
mpi_comm = None
ranks_per_node = {host_uuid: nprocs}

if MPI is not None:
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    nprocs = mpi_comm.size
    ranks_per_node = dict(Counter(mpi_comm.allgather(host_uuid)))

# INIT NCCL
nccl_comm = None

if nccl is not None:
    nccl_id = nccl.ncclGetUniqueId()
    if mpi_comm:
        nccl_id = mpi_comm.bcast(nccl_id)

    try:
        nccl_comm = nccl.ncclCommInitRank(nprocs, nccl_id, rank)
    except Exception as e:
        logger.debug("nccl init failed", exc_info=e)
        errors.append(e)

if nccl_comm:
    assert nccl
    nccl_destroy = nccl.ncclCommDestroy

    def _destory() -> None:
        assert nccl and nccl_comm
        nccl.ncclCommDestroy(nccl_comm)

    atexit.register(_destory)

# INIT CUPY
cupy_count = 0
cupy_device = None
cupy_stream = None

if cupy_driver:
    try:
        cupy_count = cupy_driver.runtime.getDeviceCount()
    except Exception as e:
        logger.debug("cupy init failed", exc_info=e)
        errors.append(e)

if cupy_count > 0:
    assert cupy_driver
    cupy_device = rank % cupy_count
    if cupy_driver.runtime.is_hip:
        os.environ["HIP_VISIBLE_DEVICES"] = str(cupy_device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cupy_device)

    from cupy.cuda import Stream as CupyStream
    cupy_driver.runtime.setDevice(cupy_device)
    cupy_stream: CupyStream = cupy_driver.get_current_stream()
    # stream_handle = stream.ptr


# INIT PYCUDA
pycuda_count = 0
pycuda_device = None
pycuda_stream = None

if pycuda_driver is not None:
    try:
        pycuda_driver.init()
        pycuda_count = pycuda_driver.Device.count()
    except Exception as e:
        logger.debug("pycuda init failed", exc_info=e)
        errors.append(e)

if pycuda_count > 0:
    assert pycuda_driver

    pycuda_device = rank % pycuda_count
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pycuda_device)

    from pycuda.driver import Stream as PycudaStream
    device = pycuda_driver.Device(pycuda_device)
    context = device.make_context()
    pycuda_stream: PycudaStream = pycuda_driver.Stream()
    # stream_handle = stream.handle

    def _destroy() -> None:
        assert context
        context.pop()

    atexit.register(_destroy)


# INIT CUDNN
cudnn_handle = None  # pyright: ignore[reportAssignmentType]

if cudnn is not None:
    try:
        cudnn_handle: int = cudnn.cudnnCreate()
    except Exception as e:
        logger.debug("cudnn init failed", exc_info=e)
        errors.append(e)

if cudnn_handle is not None:
    def _destroy() -> None:
        assert cudnn and cudnn_handle
        cudnn.cudnnDestroy(cudnn_handle)

    atexit.register(_destroy)

# INIT CUBLAS
cublas_handle: int = None  # pyright: ignore[reportAssignmentType]

if cublas is not None:
    try:
        cublas_handle: int = cublas.cublasCreate()
    except Exception as e:
        logger.debug("cublas init failed", exc_info=e)
        errors.append(e)

if cublas_handle is not None:
    def _destroy() -> None:
        assert cublas, cublas_handle
        cublas.cublasDestroy(cublas_handle)

    atexit.register(_destroy)

# SYNC CUDNN
if cudnn is not None and pycuda_stream is not None:
    cudnn.cudnnSetStream(cudnn_handle, pycuda_stream.handle)

# SYNC CUBLAS
if cublas is not None and pycuda_stream is not None:
    cublas.cublasSetStream(cublas_handle, pycuda_stream.handle)

# ENVIRONMENT
metadata = [
    {
        "time": timestamp,
        "node": host_uuid,
        "host": " ".join(platform.uname()).strip(),
        "cpu": str(cpu_affinity),
        "pycuda": str(pycuda_device),
        "cupy": str(cupy_device),
        "mpi": str(rank),
        "packages": {
            platform.python_implementation().lower(): platform.python_version(),
            **{
                pack: libdata.version(pack)
                for pack in itertools.chain.from_iterable(libdata.packages_distributions().values())
            },
        },
        "environ": dict(os.environ),
    }
]
if MPI is not None:
    metadata = MPI.COMM_WORLD.allgather(metadata[0])
