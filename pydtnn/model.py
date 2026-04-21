"""
PyDTNN model
"""
from pydtnn.utils.memory_pool import PrivateMemory, PreallocMemory
from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array, NetworkAlgEnum, ArrayShape, Parameters
from pydtnn.utils.tensor import SampleFormat, TensorFormat, format_reshape
from pydtnn.utils.performance_counter import PerformanceCounter
from pydtnn.tracers.tracer import Tracer
from pydtnn.tracers.simple_tracer_pmlib import SimpleTracerPMLib
from pydtnn.tracers.simple_tracer_gpu import SimpleTracerPycuda
from pydtnn.tracers.simple_tracer import SimpleTracer
from pydtnn.tracers.extrae_tracer import ExtraeTracer
from pydtnn.parser import PydtnnArgumentParser
from pydtnn.schedulers.scheduler import select as select_scheduler

from pydtnn.optimizers.optimizer import select as select_optimizer
from pydtnn.datasets.dataset import select as select_dataset
from pydtnn.losses.loss import Loss
from pydtnn.abstract.layerable import Layerable
from pydtnn.datasets.dataset import Dataset
from pydtnn.libs.mpi.rc import proto as PROTOCOL
from pydtnn.activations.relu import Relu
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn import utils
from pydtnn import hostname, ranks_per_node, num_gpus, nccl_comm, cudnn_handle, cublas_handle, context, stream
from pydtnn import MPI_MODULE, Cudnn_Handle_Type, Cublas_Handle_Type, drv, gpuarray, nccl, cudnn, cublas  # type: ignore (cublas exist)
import numpy as np
from collections import abc
from typing import TYPE_CHECKING, Any, Literal
from types import ModuleType
import itertools
import enum
import logging
logger = logging.getLogger(__name__)


# from warnings import filterwarnings
# filterwarnings("error")


# TODO: Check if all the elements imported here are necessary and if they are corretly set in Model's code.


if TYPE_CHECKING:
    import polyhe
else:
    try:
        import polyhe
    except Exception:
        polyhe = None

# --- CONSTANS --- #
BAR_WIDTH = 140
DEFAULT_BACH_SIZE = 64

# NOTE: mpi4py has more functions, but no typing
if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM
else:
    MPI_COMM = ModuleType

class CudnnDataType(enum.StrEnum):
    FLAOT64 = "CUDNN_DATA_DOUBLE"
    FLOAT32 = "CUDNN_DATA_FLOAT"
    INT8 = "CUDNN_DATA_INT8"
    INT32 = "CUDNN_DATA_INT32"


class Model[T: Array]:


    


