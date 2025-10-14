"""
PyDTNN model
"""

from pydtnn import crypt
from pydtnn.comm import proto as PROTOCOL
import functools
import importlib
import os
import sys
import time
from timeit import default_timer as timer
from warnings import warn
# warnings.filterwarnings("error")
from functools import cached_property

# Typing-related import
from typing import Any, TypeVar, Callable, TYPE_CHECKING, Literal
from collections.abc import Iterable
from pydtnn.tracers import SimpleTracerGPU

from types import ModuleType
from pydtnn.activations import Activation
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.tracers.tracer import Tracer
from pydtnn.datasets import Dataset
from pydtnn.losses import Loss

from tqdm import tqdm
import numpy as np

import pydtnn.metrics
from pydtnn.utils.types import Array, NetworkAlgEnum
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn import losses, metrics
from pydtnn import utils
from pydtnn.datasets import CustomDataset, get_dataset
from pydtnn.datasets.dataset import DatasetEnum
from pydtnn.lr_schedulers import get_lr_schedulers
from pydtnn.optimizers import get_optimizer
from pydtnn.parser import PydtnnArgumentParser
from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, ExtraeTracer, SimpleTracer, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.best_of import BestOf
from pydtnn.utils.memory_cache import MemoryCache
from pydtnn.utils.performance_counter import PerformanceCounter
from pydtnn.layers import Layer
import enum


# --- CUDA related imports --- #
import atexit
cuda_error_msg = list()
try:
    import pydtnn.backends.gpu.tensor_gpu
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
except Exception as e:
    gpuarray = None
    cuda_error_msg.append(f"Import: \"import pycuda.gpuarray as gpuarray\". Error: {e}")

try:
    # noinspection PyUnresolvedReferences
    import pycuda.driver as drv
    from pydtnn.backends.gpu.libs import libcudnn as cudnn
except Exception as e:
    drv = None
    cuda_error_msg.append(f"Import: \"import pycuda.driver as drv\". Error: {e}")

try:
    # noinspection PyUnresolvedReferences
    from skcuda import cublas
except Exception as e:
    cublas: ModuleType | None = None
    cuda_error_msg.append(f"Import: \"from skcuda import cublas\". Error: {e}")

# --- END CUDA related imports --- #

# --- GLOBAL VARIABLES --- #
supported_gpu: bool = False
supported_cudnn: bool = True
supported_nccl: bool = True
enable_cudnn: bool = False
# --- END GLOBAL VARIABLES --- #
try:
    from pydtnn.comm import MPI
    # noinspection PyUnresolvedReferences,PyPackageRequirements
except Exception as e:
    MPI = None

# --- CONSTANS --- #
BAR_WIDTH = 140


class ModelModeEnum(enum.Enum):
    EVALUATE = enum.auto()
    TRAIN = enum.auto()
    UNSPECIFIED = enum.auto()


DEFAULT_BACH_SIZE = 64


class LoadStoreMode(enum.Enum):
    LOAD = enum.auto()
    STORE = enum.auto()
# --- END CONSTANS --- #


# NOTE: Check "_initialize_cuda" to get the actual types.
NCCL_DataType = TypeVar("NCCL_DataType")
NCCL_Comm_Type = TypeVar("NCCL_Comm_Type")
Cudnn_Handle_Type = TypeVar("Cudnn_Handle_Type")
Cublas_Handle_Type = TypeVar("Cublas_Handle_Type")
PyCuda_Stream_Type = TypeVar("PyCuda_Stream_Type")
Cudnn_dtype = TypeVar("Cudnn_dtype")
Cudnn_Contex_Type = TypeVar("Cuda_Context")




def _layer_id_generator() -> Iterable[int]:
    """To obtain consecutive layer ids. See Layer.set_model()."""
    current_layer_id = 0
    while True:
        yield current_layer_id
        current_layer_id += 1
# --- END _layer_id_generator --- #


def ensure_model_is_runnable(method: Callable):
    @functools.wraps(method)
    def wrapper_ensure_model_is_runnable(*args, **kwargs) -> Callable:
        self: Model = args[0]
        if not self._initialized:
            self._initialize()
        are_layers = bool(self.layers)
        if not are_layers:
            warn("The model has no layers in it.", RuntimeWarning)
        elif not self.dataset:
            raise ValueError("There is no dataset and the model has layers.")
        return method(*args, **kwargs)

    return wrapper_ensure_model_is_runnable
# --- END ensure_model_is_runnable --- #


# NOTE: can not specify a particular module
type MPI_MODULE = ModuleType

# NOTE: mpi4py has more functions, but no typing
if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM
else:
    MPI_COMM = ModuleType


def _initilize_communications(parallel: str) -> tuple[None, None] | tuple[MPI_MODULE, MPI_COMM]:
    match parallel:
        case "sequential":
            return (None, None)
        case "data":
            if not MPI:
                raise SystemExit("Please, install mpi4py to allow parallel MPI execution!")
            return (MPI, MPI.COMM_WORLD)
        case _:
            raise SystemExit(f"Parallel option '{parallel}' not recognized.")
# --- END _initilize_communications --- #


def _set_execution_attributes(self: "Model", comm: MPI_COMM | None, shared_storage: bool) -> None:
    self.rank_weight = 1.0
    self.comm_rank = self.rank = 0
    self.comm_size = self.nprocs = 1
    if comm:
        self.comm_rank = comm.Get_rank()
        self.comm_size = comm.Get_size()
        if shared_storage:
            self.rank = self.comm_rank
            self.nprocs = self.comm_size
        # else: Nothing each rank is independant
    self.comm_groups = self.comm_size // self.nprocs
# --- END _set_execution_attributes --- #


def _initilize_and_get_tracer(tracer_output: str, tracing: bool, comm: ModuleType, enable_gpu: bool,
                              tracer_pmlib_server: str, tracer_pmlib_port: int, tracer_pmlib_device: str) -> Tracer:

    if tracer_output == "":
        tracer = ExtraeTracer(tracing)
    else:
        if enable_gpu:
            tracer = SimpleTracerGPU(tracing, tracer_output, comm)
        else:
            if tracer_pmlib_device != "":
                from pydtnn.tracers import SimpleTracerPMLib
                tracer = SimpleTracerPMLib(tracing, tracer_output, comm,
                                           tracer_pmlib_server, tracer_pmlib_port, tracer_pmlib_device)
            else:
                tracer = SimpleTracer(tracing, tracer_output, comm)
    return tracer
# --- END _initilize_and_get_tracer --- #


def _initialize_cuda(self: "Model", comm: ModuleType, comm_rank: int, rank: int, nprocs: int,
                     gpus_per_node: int, parallel: str, dtype: np.dtype,
                     enable_nccl: bool, tracer: SimpleTracer, gpudirect: bool) -> None:

    global supported_cudnn, supported_nccl
    supported_cudnn = True
    supported_nccl = True

    device_id: int = comm_rank % drv.Device.count()
    drv.init()
    context: Cudnn_Contex_Type = drv.Device(device_id).make_context()
    # context:int = drv.Device(device_id).retain_primary_context()

    atexit.register(context.pop)

    nccl_type = None
    nccl_comm = None

    if not gpudirect and enable_nccl:
        raise RuntimeError("It is necessary to have gpudirect active to work with NCCL.")

    if comm and enable_nccl:
        try:
            from pydtnn.backends.gpu.libs import libnccl as nccl
        except Exception as e:
            supported_nccl = False
            msg = "Please, install nccl to be able to use NVIDIA NCCL inter-GPU communications!"
            raise SystemExit(msg) from None

        nccl_types = {np.float64: nccl.DataType.Float64,
                      np.float32: nccl.DataType.Float32,
                      np.int8: nccl.DataType.Int8,
                      np.int32: nccl.DataType.Int32}

        nccl_type: NCCL_DataType = nccl_types.get(dtype, nccl.DataType.Float32)

        hostname = MPI.Get_processor_name()

        hosts_data = comm.allgather([rank, hostname])
        # Build a dictionary hostname : [ranks_in_host]
        #   { "host1": [0, 1], "host2": [2, 3] }
        hosts = {}
        for r, h in hosts_data:
            # noinspection PyTypeChecker
            hosts.setdefault(h, []).append(r)
        if parallel == "data":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % gpus_per_node)
        # Check that no more processes than GPUs per node are used
        for host, ranks_in_host in hosts.items():
            if len(ranks_in_host) > gpus_per_node:
                raise SystemExit("Not able to run more processes than GPUs per node!")

        nccl_id = comm.bcast(nccl.ncclGetUniqueId() if comm_rank == 0 else None)
        nccl_comm: NCCL_Comm_Type = nccl.ncclCommInitRank(nprocs, nccl_id, rank)

    cudnn_handle: Cudnn_Handle_Type = cudnn.cudnnCreate()
    cublas_handle: Cublas_Handle_Type = cublas.cublasCreate()
    stream: PyCuda_Stream_Type = drv.Stream()
    cublas.cublasSetStream(cublas_handle, stream.handle)
    cudnn.cudnnSetStream(cudnn_handle, stream.handle)

    cudnn_types = {np.float64: "CUDNN_DATA_DOUBLE",
                   np.float32: "CUDNN_DATA_FLOAT",
                   np.int8: "CUDNN_DATA_INT8",
                   np.int32: "CUDNN_DATA_INT32"}

    cudnn_type: str = cudnn_types.get(dtype, "CUDNN_DATA_FLOAT")

    cudnn_dtype: Cudnn_dtype = cudnn.cudnnDataType[cudnn_type]
    tracer.set_default_stream(stream)

    self.nccl_type = nccl_type
    self.nccl_comm = nccl_comm
    self.cudnn_handle = cudnn_handle
    self.cublas_handle = cublas_handle
    self.stream = stream
    self.cudnn_dtype = cudnn_dtype
# --- END _initialize_cuda --- #


def _set_data_format(tensor_format: Literal["AUTO", "NCHW", "NHWC"] = "AUTO", enable_cudnn: bool = False) -> PYDTNN_TENSOR_FORMAT:
    match tensor_format:
        case "AUTO":
            tensor_format = PYDTNN_TENSOR_FORMAT.NCHW if enable_cudnn else PYDTNN_TENSOR_FORMAT.NHWC
        case "NCHW":
            tensor_format = PYDTNN_TENSOR_FORMAT.NCHW
        case "NHWC":
            tensor_format = PYDTNN_TENSOR_FORMAT.NHWC
        case _:
            raise NotImplementedError(f"\'{tensor_format}\' is not supported.")
    return tensor_format
# --- END _set_data_format --- #


def _calculate_batch_size(batch_size: int | None, global_batch_size: int | None, comm_size: int) -> int:

    if batch_size and global_batch_size:
        raise SystemExit("Can not define 'batch_size' and 'global_batch_size' simultaneously")

    if global_batch_size:
        # NOTE: Using comm_size instead of nprocs might not be appropriate,
        #       as it differs to how global_batch_size is defined elsewhere,
        #       but for now it just a parser option difference that helps testing
        _batch_size = global_batch_size // comm_size
    elif batch_size:

        _batch_size = batch_size
    else:
        _batch_size = DEFAULT_BACH_SIZE

    if _batch_size < 1:
        raise SystemExit(f"'batch_size' ({_batch_size}) too small or too many processes (num processes: {comm_size})")

    return _batch_size
# --- END _calculate_batch_size --- #

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
class Model[T: Array]:
    """
    PyDTNN Model
    """

    def __init__(self, parallel: Literal["sequential", "data"] = "sequential", use_blocking_mpi: bool = False, enable_gpu: bool = False,
                 enable_gpudirect: bool = False, enable_nccl: bool = False, dtype: np.dtype = np.float32, tracing: bool = False,
                 tracer_output: str = "", tracer_pmlib_server: str = "127.0.0.1", tracer_pmlib_port: int = 6526,
                 tracer_pmlib_device: str = "", **kwargs):
        # Attributes related to the given arguments
        self.parallel: bool = parallel
        self.blocking_mpi: bool = use_blocking_mpi
        global enable_cudnn
        self.enable_gpu = enable_cudnn = self.enable_cudnn = enable_gpu
        self.gpudirect: bool = enable_gpudirect
        self.enable_nccl: bool = enable_nccl
        self.dtype: np.dtype = dtype

        self.rank_weight: int = -1
        self.comm_rank: int = -1
        self.comm_size: int = -1
        self.rank: int = -1
        self.nprocs: int = -1
        self.comm_groups: int = -1

        self.num_real_batches: int = -1
        self._sync_x_y = self._sync_x_y_gpu if self.enable_gpu else self._sync_x_y_cpu

        self.nparams: int = 0  # NOTE: Model's total number of params

        # The following attributes will be initilized later only if "enable_cudnn" is True.
        self.nccl_type: NCCL_DataType | None = None
        self.nccl_comm: NCCL_Comm_Type | None = None
        self.cudnn_handle: Cudnn_Handle_Type | None = None
        self.cublas_handle: Cublas_Handle_Type | None = None
        self.stream: PyCuda_Stream_Type | None = None
        self.cudnn_dtype: Cudnn_dtype | None = None
        self.input_shape: tuple[int, ...] = None
        self.output_shape: tuple[int, ...] = None

        # Get default values from parser and update them from the received kwargs
        self.kwargs: dict[str, Any] = PydtnnArgumentParser().get_default_values()
        self.kwargs.update(kwargs)

        # Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
        #   NOTE: The following parameters come from "Parser"
        self.steps_per_epoch: int = self.kwargs['steps_per_epoch']
        self.cpu_speed: float = self.kwargs['cpu_speed']
        self.memory_bw: float = self.kwargs['memory_bw']
        self.network_bw: float = self.kwargs['network_bw']
        self.network_lat: float = self.kwargs['network_lat']
        self.network_alg: NetworkAlgEnum = NetworkAlgEnum(self.kwargs['network_alg'].lower())
        self.loss_func_name: str = self.kwargs['loss_func_name']
        self.num_epochs: int = self.kwargs['num_epochs']
        self.model_sync_freq: int = self.kwargs['model_sync_freq']
        self.final_model_sync: bool = self.kwargs['final_model_sync']
        self.test_as_validation: bool = self.kwargs['test_as_validation']
        self.validation_split: float = self.kwargs['validation_split']
        self.use_synthetic_data: bool = self.kwargs['use_synthetic_data']
        self.dataset_train_path: str = self.kwargs['dataset_train_path']
        self.dataset_test_path: str = self.kwargs['dataset_test_path']
        self.enable_best_of: bool = self.kwargs['enable_best_of']
        self.enable_conv_i2c: bool = self.kwargs['enable_conv_i2c']
        self.enable_conv_winograd: bool = self.kwargs['enable_conv_winograd']
        self.enable_conv_gemm: bool = self.kwargs['enable_conv_gemm']
        self.enable_conv_direct: bool = self.kwargs['enable_conv_direct']
        self.evaluate_only: bool = self.kwargs['evaluate_only']
        self.evaluate_on_train: bool = self.kwargs['evaluate_on_train']
        self.profile: bool = self.kwargs['profile']
        self.history_file: str = self.kwargs['history_file']
        self.model_sync_min_avail: int = self.kwargs['model_sync_min_avail']
        self.dataset_name: str = self.kwargs['dataset_name']
        self.shared_storage: bool = self.kwargs["shared_storage"]
        self.encryption_name: str = self.kwargs["encryption_name"]
        self.flip_images: bool = self.kwargs["flip_images"]
        self.crop_images: bool = self.kwargs["crop_images"]
        self.crop: bool = self.kwargs["crop"]
        self.crop_dimension: int = self.kwargs["crop_dimension"]
        self.resize: bool = self.kwargs["resize"]
        self.resize_dimension: int = self.kwargs["resize_dimension"]
        self.flip_images_prob: float = self.kwargs["flip_images_prob"]
        self.crop_images_size: int = self.kwargs["crop_images_size"]
        self.crop_images_prob: float = self.kwargs["crop_images_prob"]
        self.initial_model_sync: bool = self.kwargs["initial_model_sync"]
        self.dataset_percentage: float = self.kwargs["dataset_percentage"]
        # ---
        use_mpi_buffers: bool = self.kwargs["use_mpi_buffers"]

        match use_mpi_buffers:
            case None:
                self.use_mpi_buffers: bool = PROTOCOL is None
            case bool():
                self.use_mpi_buffers: bool = use_mpi_buffers
            case _:
                raise SystemExit(f"MPI buffers option '{use_mpi_buffers}' not recognized.")

        # Set MPI and comm
        self.MPI, self.comm = _initilize_communications(parallel=parallel)

        # Execution attributes
        _set_execution_attributes(self, comm=self.comm, shared_storage=self.shared_storage)

        # Set tracer
        self.tracer = _initilize_and_get_tracer(tracer_output=tracer_output, tracing=tracing, comm=self.comm, enable_gpu=enable_gpu,
                                                tracer_pmlib_server=tracer_pmlib_server, tracer_pmlib_port=tracer_pmlib_port,
                                                tracer_pmlib_device=tracer_pmlib_device)

        # Set performance counter
        self.perf_counter = PerformanceCounter()

        # Layers' attributes
        self.layers: list[Layer | Activation] = []
        self.layer_id: int = _layer_id_generator()

        # Matmul
        self.matmul = utils.matmul

        # Set current mode to unspecified
        self.mode: ModelModeEnum = ModelModeEnum.UNSPECIFIED

        # Memory cache optimization
        self.enable_memory_cache: bool  # NOTE: This parameter comes from "Parser"
        if self.enable_memory_cache:
            MemoryCache.enable()
        else:
            MemoryCache.disable()

        global cuda_error_msg

        # Cuda
        if self.enable_cudnn:
            if gpuarray and drv and cublas:
                self.gpus_per_node: int  # NOTE: This parameter comes from "Parser"
                _initialize_cuda(self, comm=self.comm, comm_rank=self.comm_rank,
                                 rank=self.rank, nprocs=self.nprocs,
                                 gpus_per_node=self.gpus_per_node,
                                 parallel=self.parallel, dtype=self.dtype,
                                 enable_nccl=self.enable_nccl, tracer=self.tracer,
                                 gpudirect=self.gpudirect)
            else:
                raise ImportError("\n".join(cuda_error_msg))
        else:
            cuda_error_msg = None  # If CUDA is not going to be used, then the import errors should be deleted (or mark to be deleted).

        # Data format
        # NOTE: self.kwargs["tensor_format"] value comes from Parser.
        self.tensor_format: PYDTNN_TENSOR_FORMAT = _set_data_format(tensor_format=self.kwargs["tensor_format"], enable_cudnn=self.enable_cudnn)

        # Disable BestOf globally if not enabled
        if self.kwargs['enable_best_of'] is False:
            # NOTE: comes from "Parser"
            BestOf.use_always_the_first_alternative()

        self.batch_size = _calculate_batch_size(batch_size=self.kwargs['batch_size'],  # NOTE: This parameters comes from "Parser"
                                                global_batch_size=self.kwargs['global_batch_size'],  # NOTE: This parameters comes from "Parser"
                                                comm_size=self.comm_size)

        # Attributes that will be properly initialized elsewhere
        self.y_batch: T = None
        self.history: dict[str, list[np.ndarray]] = None
        self.loss_func: Loss = None
        self.metrics_funcs: list[metrics.Metric] = None
        self.loss_and_metrics: list[str]  # Is a list with the name of the loss function and the metrics's names.
        self.total_metrics: np.ndarray
        # ---

        # Encryption
        if self.encryption_name:
            self.crypt = self._init_crypt(self.encryption_name)

        else:
            self.crypt = None

        self.weights_and_bias_filename: str  # NOTE: This parameter comes from "Parser"
        # Load weights and bias
        if self.weights_and_bias_filename:
            self.load_weights_and_bias(self.weights_and_bias_filename)
        # Dataset
        if self.dataset_name:
            self.dataset: Dataset = get_dataset(self)

        # Optimizers and LRSchedulers
        # NOTE: 'self.kwargs["learning_rate_scaling"]' comes from "Parser"
        if self.kwargs["learning_rate_scaling"]:
            # using comm_size instead of nprocs might not be appropriate,
            # as it differs to how learning_rate is defined elsewhere,
            # but for now it just a parser option difference that helps testing
            self.learning_rate: float = self.kwargs["learning_rate"] / self.comm_size

        self.optimizer = get_optimizer(self)
        self.lr_schedulers = get_lr_schedulers(self)
        # Metrics list
        self.metrics: str  # NOTE: This variable comes from the Parser.
        self.metrics_list: list[metrics.Metric] = [m for m in self.metrics.replace(" ", "").split(",")]
        # Private attributes
        self._evaluate_round: int = 0
        self._initialized: bool = False
        # Read the model (must be the last action, as it calls self._initialize() if there is a model)
        self.model_name: str | None = self.kwargs.get("model_name")
        if self.model_name:
            self._read_model(self.model_name)
        # Syncronization parameters
        self.model_sync_alg: str  # NOTE: This parameter come from Parser.
        self.model_sync_participation: str  # NOTE: This parameter come from Parser.
        if self.model_sync_alg not in {"avg", "wavg", "invwavg"}:
            raise SystemExit(f"Process weight option '{self.proc_weight}' not recognized.")
        if self.model_sync_participation not in {"all", "avail2all"}:
            raise SystemExit(f"Process weight option '{self.proc_weight}' not recognized.")
    # --- END __init__ --- #

    #@property
    #@cache
    @cached_property
    def empty_x(self) -> TensorGPU:
        # NOTE: it's necessary to first set the size to 1 and then make a slice of 0 because otherwise it throws different exceptions related to trying to fill nothing or due not reserving the GPU's memory.
        empty_x = gpuarray.empty((1, *self.dataset.input_shape), self.dtype)
        return TensorGPU(empty_x[:0], self.tensor_format, self.cudnn_dtype)

    #@property
    #@cache
    @cached_property
    def empty_y_tag(self) -> TensorGPU:
        # NOTE: it's necessary to first set the size to 1 and then make a slice of 0 because otherwise it throws different exceptions related to trying to fill nothing or due not reserving the GPU's memory.
        empty_y_tag = gpuarray.empty((1, *self.dataset.output_shape), self.dtype)
        return TensorGPU(empty_y_tag[:0], self.tensor_format, self.cudnn_dtype)

    @property
    def dataset_path(self) -> str:
        """Raw dataset path with rank substituted"""
        return utils.string_substitute(self.kwargs["dataset_path"], rank=self.comm_rank)
    # --- END dataset_raw_path --- #

    def __getattr__(self, item) -> Any:
        try:
            return self.kwargs[item]
        except KeyError:
            raise AttributeError(f"Model object has no attribute '{item}'!") from None
    # --- End __getattr__ --- #

    def _init_crypt(self, encryption_name: str) -> crypt.Context:
        """Inizialize encryption context"""
        try:
            module = importlib.import_module(f"pydtnn.crypt.{encryption_name}")
        except Exception as exc:
            import traceback
            print(traceback.print_exception(exc))
            sys.exit(-1)

        if self.comm_rank == 0:
            crypt = module.Context(
                poly_degree=self.encryption_poly_degree,
                global_scale=self.encryption_global_scale,
                security_level=self.encryption_security_level
            )

        if self.comm:
            crypt = self.comm.bcast(crypt if self.comm_rank == 0 else None)

        assert crypt is not None
        if self.enable_nccl:
            warn("If NCCL is active, encryption is disabled", RuntimeWarning)

        return crypt

    def _read_model(self, model_name: str) -> None:
        try:
            model_module = importlib.import_module(f"pydtnn.models.{model_name}")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            sys.exit(-1)

        # NOTE: Dataset is always in NCHW
        c, h, w = self.dataset.input_shape
        input_shape = (h, w, c) if self.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC else (c, h, w)
        if len(input_shape) != 3:
            warn(f"Input layer does not have 3 dimensions ({input_shape}), it may cause issues!", RuntimeWarning)
        launch_shape_warning = len(input_shape) == 3 and not (input_shape[0] > input_shape[2]) if self.tensor_format is PYDTNN_TENSOR_FORMAT.NHWC \
            else len(input_shape) == 3 and not (input_shape[0] < input_shape[1])
        if launch_shape_warning:
            warning_text = f"Input layer shape {input_shape} may not be in {self.tensor_format} format, regardless of model format! "
            warn(warning_text, RuntimeWarning)
            warning_text = None
        output_shape = tuple(self.dataset.output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape

        layers = getattr(model_module, f"create_{model_name}")(input_shape, output_shape)
        self.add_layers(layers)

        self._initialize()
    # --- END _read_model --- #

    def show(self) -> None:
        bfp = np.dtype(self.dtype).itemsize
        line = "+-------+--------------------------+---------+---------------+-------------------" \
               "+-------------------------------------+"
        head = "| Layer |           Type           | #Params | Output shape  |   Weights shape   " \
               "|             Parameters              |"
        print(line)
        print(head)
        for layer in self.layers:
            print(line)
            layer.show()
        print(line)
        print(f"|{'':^7s} {'Total parameters':^26s} {self.nparams:^9d} {utils.convert_size_bytes(self.nparams * bfp):^15s} "
              f"{'':19s} {'':37s}|")
        print(line)
    # --- END show --- #

    def print_in_convdirect_format(self) -> None:
        line = "#l\tkn\two\tho\tt\tkh\tkw\tci\twi\thi"
        print(line)
        for layer in self.layers:
            layer.print_in_convdirect_format()
    # --- END print_in_convdirect_format --- #

    def add(self, layer: Layer | Activation) -> None:
        layer.set_model(self)

        if layer.id > 0:
            prev_shape = self.layers[-1].shape
            y = self.layers[-1].y
        else:
            prev_shape = ()
            y = None

        layer.initialize(prev_shape, y)

        self.nparams += layer.nparams
        self.layers.append(layer)

        if layer.act:
            self.add(layer.act())
    # --- END add --- #

    def add_layers(self, list_layers: list[Layer | Activation]) -> None:
        for layer in list_layers:
            self.add(layer)
    # --- END add_layers ---

    def get_all_layers(self, from_layers: list[Layer | Activation] | None = None) -> list[Layer | Activation]:
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            this_recursion_layers += self.get_all_layers(layer.children)
        return this_recursion_layers
    # --- dataset_raw_path ---

    def _apply_layer_fusion(self, bn_relu=False, conv_relu=False, conv_bn=False, conv_bn_relu=False):
        """ Apply layer fusion in a recursive manner """

        def __layer_fusion(layers: list[Layer], bn_relu=False, conv_relu=False,
                           conv_bn=False, conv_bn_relu=False):
            fused_layers: list[Layer] = []
            for i, curr_layer in enumerate(layers):
                # if i > 0: print(i, curr_layer.canonical_name, fused_layers[-1].canonical_name)
                if curr_layer.is_block_layer:
                    for j, p in enumerate(curr_layer.paths):
                        curr_layer.paths[j] = __layer_fusion(p, bn_relu, conv_relu, conv_bn, conv_bn_relu)
                elif conv_bn_relu and len(fused_layers) > 1 and \
                        curr_layer.canonical_name == "Relu" and \
                        fused_layers[-1].canonical_name == "BatchNormalization" and \
                        fused_layers[-2].canonical_name == "Conv2D":
                    backend = "gpu" if self.enable_cudnn else "cpu"
                    fused_layer = getattr(importlib.import_module(f"pydtnn.backends.{backend}.layers"),
                                          fused_layers[-2].canonical_name +
                                          fused_layers[-1].canonical_name +
                                          type(curr_layer).__name__)
                    if fused_layers[-2].forward.__name__ in fused_layer.__dict__:  # or self.enable_best_of:
                        bn_layer = fused_layers.pop()
                        cv_layer = fused_layers.pop()
                        print("Fusing %03d_%s + %03d_%s + %03d_%s..." % (cv_layer.id, type(cv_layer).__name__,
                                                                         bn_layer.id, type(bn_layer).__name__,
                                                                         curr_layer.id, type(curr_layer).__name__))
                        curr_layer = fused_layer(from_parent=cv_layer, from_parent2=bn_layer)
                        curr_layer.initialize(from_parent_dict=cv_layer.__dict__)
                elif (conv_relu or conv_bn) and len(fused_layers) > 0 and \
                        (curr_layer.canonical_name == "Relu" or
                         curr_layer.canonical_name == "BatchNormalization") and \
                        fused_layers[-1].canonical_name == "Conv2D" and \
                        not (conv_bn_relu and i + 1 < len(layers) and layers[i + 1].canonical_name == "Relu"):
                    backend = "gpu" if self.enable_cudnn else "cpu"
                    fused_layer = getattr(importlib.import_module(f"pydtnn.backends.{backend}.layers"),
                                          fused_layers[-1].canonical_name +
                                          type(curr_layer).__name__)
                    if fused_layers[-1].forward.__name__ in fused_layer.__dict__:  # or self.enable_best_of:
                        prev_layer = fused_layers.pop()
                        print("Fusing %03d_%s + %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                                curr_layer.id, type(curr_layer).__name__))
                        curr_layer = fused_layer(from_parent=prev_layer, from_parent2=curr_layer)
                        curr_layer.initialize(from_parent_dict=prev_layer.__dict__)
                elif bn_relu and len(fused_layers) > 0 and \
                        curr_layer.canonical_name == "Relu" and \
                        fused_layers[-1].canonical_name == "BatchNormalization":
                    prev_layer = fused_layers.pop()
                    print("Fusing %03d_%s + %03d_%s ..." % (prev_layer.id, type(prev_layer).__name__,
                                                            curr_layer.id, type(curr_layer).__name__))
                    curr_layer = getattr(importlib.import_module("pydtnn.layers"),
                                         prev_layer.canonical_name +
                                         curr_layer.canonical_name)(from_parent=prev_layer)
                fused_layers.append(curr_layer)
            return fused_layers

        if not self.enable_cudnn and (bn_relu or conv_relu or conv_bn, conv_bn_relu):
            self.layers = __layer_fusion(self.layers, bn_relu, conv_relu, conv_bn, conv_bn_relu)
    # --- END _apply_layer_fusion --- #

    def _initialize(self):
        if self._initialized:
            return
        # NOTE: all this "[keyword]" in self.kwargs.get([keyword]) come from Parser
        self._apply_layer_fusion(self.kwargs.get("enable_fused_bn_relu"), self.kwargs.get("enable_fused_conv_relu"),
                                 self.kwargs.get("enable_fused_conv_bn"), self.kwargs.get("enable_fused_conv_bn_relu"))
        # TODO/FIXME: Pass the loss' class as a parameter insted of get it here.
        self.loss_func = losses.switch_losses(self.loss_func_name)(shape=(self.batch_size, *self.layers[-1].shape), model=self)
        self.metrics_funcs = [getattr(metrics, m)(shape=(self.batch_size, *self.layers[-1].shape), model=self) for m in
                              self.metrics_list]
        self.loss_and_metrics = [self.loss_func_name] + self.metrics_list
        self.total_metrics = np.array([0] + [0 for func in self.metrics_funcs], dtype=self.dtype)
        self.tracer.define_event_types(self)
        self._initialized = True

        self.optimizer.initialize(self.get_all_layers(self.layers))
    # --- End _initialize --- #

    def load_store_path(self, layers: list[Layer], d: dict[str, np.ndarray], mode: LoadStoreMode) -> None:
        """
        Method to load and store the weigths and biases.

        Args:
            layers: the list of the layers.
            d: The dictionary of layers (keys) with their respective Weights and Biases (values), that are numpy's ndarray.
            mode: Values from the enum "LoadStoreMode".
                - "LoadStoreMode.LOAD" (that is "load") mode loads the data from "d" into the Model.
                - "LoadStoreMode.STORE" (that is "store") mode stores the data from the Model into "d".
        """
        for layer in layers:
            name = layer.canonical_name
            if name in ["AdditionBlock", "ConcatenationBlock"]:
                for path in layer.paths:
                    self.load_store_path(path, d, mode)
            else:
                grad_vars = [g for g in layer.grad_vars] + \
                    (["running_var", "running_mean"] if name == "BatchNormalization" else [])
                for key in grad_vars:
                    base = f"{layer.id}_{name}_{key}"
                    if mode is LoadStoreMode.LOAD and base not in d:
                        print(f"Could not find '{base}' for layer '{name}' in file!")
                        continue
                    match mode:
                        case LoadStoreMode.LOAD:
                            if self.enable_cudnn:
                                # NOTE: getattr(layer, key): TensorGPU, ary: gpuarray
                                ary = getattr(layer, key).ary
                                ary.set(d[base].reshape(ary.shape))
                            else:
                                setattr(layer, key, d[base])
                        case LoadStoreMode.STORE:
                            if self.enable_cudnn:
                                # NOTE: getattr(layer, key): TensorGPU
                                d[base] = getattr(layer, key).ary.get()
                            else:
                                d[base] = getattr(layer, key)
                        case _:
                            raise NotImplementedError(f"Function: \"load_store_path\". mode:\"{mode}\"")
    # --- END load_store_path --- #

    def load_weights_and_bias(self, filename: str) -> None:
        """
        ARGS:
            filename: Path to the file with the weights and biases to load.
        """
        d = np.load(filename)
        self.load_store_path(self.layers, d, LoadStoreMode.LOAD)

    def store_weights_and_bias(self, filename: str, compress=True) -> None:
        """
        ARGS:
            filename: Path to the file were the weights and biases will be stored.
        """
        d = {}
        self.load_store_path(self.layers, d, LoadStoreMode.STORE)
        if compress:
            np.savez_compressed(filename, **d)
        else:
            np.savez(filename, **d)

    def calculate_time(self) -> np.ndarray:
        # Total elapsed_time, Comp elapsed_time, Memo elapsed_time, Net elapsed_time
        total_time: np.ndarray = np.zeros((4,), dtype=np.float32)

        first_layer = 1  # Remember: The "Input" layer (the 0th layer) forward and backward function do nothing, so we skip it.
        last_layer = len(self.layers) - 1
        # Forward pass (FP)
        for layer in range(first_layer, last_layer + 1):
            total_time += self.layers[layer].fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in range(last_layer, first_layer - 1, -1):
                total_time += self.layers[layer].bwd_time

            # Weight update (WU)
            for layer in range(last_layer, first_layer - 1, -1):
                if self.comm and self.layers[layer].weights.size > 0:
                    total_time += allreduce_time(self.layers[layer].weights.size + self.layers[layer].biases.size,
                                                 self.cpu_speed, self.network_bw, self.network_lat,
                                                 self.network_alg, self.nprocs, self.dtype)
        else:
            total_time_iar: int = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in range(last_layer, -1, -1):
                total_time += self.layers[layer].bwd_time
                if self.comm and self.layers[layer].weights.size > 0:
                    time_iar = allreduce_time(self.layers[layer].weights.size + self.layers[layer].biases.size,
                                              self.cpu_speed, self.network_bw, self.network_lat,
                                              self.network_alg, self.nprocs, self.dtype)
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time
    # --- END calculate_time --- #

    def _compute_metrics_funcs[S: Array](self, y_pred: S, y_targ: S, loss: float, blocking=True, comm=True) -> tuple[np.ndarray, None] | tuple[None, Any]:
        loss_req: T | None = None
        _losses: T | None

        if y_targ.shape[0] > 0:
            if self.enable_cudnn:
                _losses = np.array([loss] + [func(y_pred.ary, y_targ.ary) for func in self.metrics_funcs], dtype=self.dtype)
            else:
                _losses = np.array([loss] + [func(y_pred, y_targ) for func in self.metrics_funcs], dtype=self.dtype)
        else:
            _losses = self.total_metrics.copy()
            _losses[0] = loss

        if self.comm is not None and comm:
            assert MPI

            _losses /= self.comm_size
            if blocking:
                self.comm.Allreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
            else:
                loss_req = self.comm.Iallreduce(MPI.IN_PLACE, _losses, op=MPI.SUM)
        else:
            if blocking:
                pass
            else:
                raise NotImplementedError("can not compute metrics non-blocking locally")

        return _losses, loss_req
    # --- END _compute_metrics_funcs --- #

    def _update_running_average(self, curr: np.ndarray, total: np.ndarray, count: np.ndarray,
                                batch_size: int, prefix="") -> tuple[np.ndarray, np.ndarray, str]:
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(self.loss_and_metrics)):
            loss_str = pydtnn.metrics.metric_format.get(self.loss_and_metrics[c], self.loss_and_metrics[c])
            string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string
    # --- END _update_running_average --- #

    def _sync_x_y(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[T, T]:
        raise TypeError("Please, use the cpu or gpu version.")
    # --- _sync_x_y --- #

    def _sync_x_y_cpu(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.optimizer.num_real_batches = self.num_real_batches = x_batch.shape[0]
        x_batch = np.asarray(x_batch, dtype=self.dtype, order='C', copy=None)
        y_batch = np.asarray(y_batch, dtype=self.dtype, order='C', copy=None)
        return x_batch, y_batch
    # --- _sync_x_y_cpu --- #

    def _sync_x_y_gpu(self, x_batch: np.ndarray, y_batch: np.ndarray) -> tuple[TensorGPU, TensorGPU] | tuple[None, None]:

        # NOTE: in CUDA it's necessary to always have batches of the same size.
        local_batch_size = x_batch.shape[0]

        self.optimizer.num_real_batches = self.num_real_batches = local_batch_size
        if local_batch_size != 0:
            if local_batch_size != self.batch_size:
                # NOTE: if x_batch is empty (local_batch_size == 0), this will mean the end of the loop where this function is called.
                num_repetitions = ceil(self.batch_size / local_batch_size)
                x_batch = np.repeat(x_batch, num_repetitions, axis=0)[:self.batch_size]
                y_batch = np.repeat(y_batch, num_repetitions, axis=0)[:self.batch_size]
            # else: The batch has the right shape ==> Nothing to do.

            self.layers[0].y.ary.set(x_batch)
            self.y_batch.ary.set(y_batch)
            x, y_targ = self.layers[0].y, self.y_batch
        else:
            x, y_targ = self.empty_x, self.empty_y_tag

        return x, y_targ
    # --- _sync_x_y_gpu --- #

    # TODO: Modify the method's name.

    def _weight_update(self, gradient=True, blocking=True):
        first_layer = 1  # Remember: The "Input" layer (the 0th layer) forward and backward function do nothing, so we skip it.
        last_layer = len(self.layers) - 1
        if blocking:
            for i in range(last_layer, first_layer - 1, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT,
                                       self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW)
                self.layers[i].reduce_weights_sync(gradient=gradient)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        else:
            for i in range(last_layer, first_layer - 1, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT,
                                       self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW)
                self.layers[i].reduce_weights_async(gradient=gradient)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            for i in range(last_layer, first_layer - 1, -1):
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT],
                                        [self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.WAIT_DW,
                                        self.layers[i].id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.OPS_ALLREDUCE_DW])
                self.layers[i].wait_allreduce_async(gradient=gradient)
                self.tracer.emit_nevent([PYDTNN_MDL_EVENT, PYDTNN_OPS_EVENT], [PYDTNN_EVENT_FINISHED, PYDTNN_EVENT_FINISHED])
    # --- END _weight_update --- #

    @ensure_model_is_runnable
    def train_dataset(self, bar_width=BAR_WIDTH) -> dict[str, list[np.ndarray]]:
        # If working with CUDA, self.y_batch must be in a GPU's data structure.
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)

        self.history = {lm: [] for lm in (self.loss_and_metrics + [f"val_{m}" for m in self.loss_and_metrics])}

        self.comm_nsamples = list(zip(*self.comm.allgather(self.dataset._nsamples) if self.comm else [self.dataset._nsamples]))

        terminate = False  # True: ends the following loop.
        global_terminate = False

        model_sync_count = 0
        train_batches_min = min(self.comm_nsamples[DatasetEnum.TRAIN]) / (self.batch_size * self.nprocs)
        val_batches_min = min(self.comm_nsamples[DatasetEnum.VAL]) / (self.batch_size * self.nprocs)

        for epoch in range(self.num_epochs):
            train_batch_generator, val_batch_generator = self.dataset.get_train_val_generator()
            sync_epoch = False

            train_total_loss, train_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            val_total_loss, val_batch_count = np.zeros(len(self.loss_and_metrics)), 0

            if self.comm_rank == 0:
                string = ""
                fmt = "%%%dd" % (len(str(self.num_epochs)))
                epoch_string = "Epoch %s/%s" % (fmt, fmt)
                pbar = tqdm(total=self.dataset.train_nsamples, ncols=bar_width,
                            ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                            desc=epoch_string % (epoch + 1, self.num_epochs), unit=" samples")

            for lr_sched in self.lr_schedulers:
                lr_sched.on_epoch_begin(self, self.rank)

            # --- TRAIN --- #
            for i_batch, (x_batch, y_batch, batch_size) in enumerate(train_batch_generator):
                if terminate:
                    x_batch = x_batch[:0]
                    y_batch = y_batch[:0]

                local_batch_size = x_batch.shape[0]
                sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)

                if sync_model:
                    sync_epoch = True

                if model_sync_count == 0 and not self.initial_model_sync:
                    sync_model = False

                model_sync_count += 1

                if i_batch >= train_batches_min and sync_model:
                    rank_mask = self.comm.allgather(min(1, local_batch_size)) if self.comm else [min(1, local_batch_size)]
                else:
                    rank_mask = [1] * self.comm_size
                rank_avail = sum(rank_mask)

                if rank_avail <= 0 or global_terminate:
                    break

                if rank_avail < self.model_sync_min_avail:
                    sync_model = False

                self.rank_weight = self._compute_rank_weight(rank_mask, DatasetEnum.TRAIN)

                tic = timer()
                train_batch_loss = self._train_batch(x_batch, y_batch, sync_model=sync_model)
                toc = timer()

                if local_batch_size <= 0:
                    if self.comm_rank == 0:
                        pbar.set_postfix_str(s=f"{string}, waiting…", refresh=True)
                    continue

                train_total_loss, train_batch_count, string = \
                    self._update_running_average(train_batch_loss, train_total_loss,
                                                 train_batch_count, batch_size)
                if self.comm_rank == 0:
                    # noinspection PyUnboundLocalVariable
                    pbar.set_postfix_str(s=string, refresh=True)
                    pbar.update(batch_size)
                    self.perf_counter.add_training_time_and_batch_size(epoch, toc - tic, batch_size)

            if self.comm_rank == 0:
                train_string = string
                for c in range(len(self.loss_and_metrics)):
                    self.history[self.loss_and_metrics[c]].append(train_total_loss[c])

            # ----------- #
            # --- VAL --- #
            # ----------- #
            for i_batch, (x_batch, y_batch, batch_size) in enumerate(val_batch_generator):
                if terminate:
                    x_batch = x_batch[:0]
                    y_batch = y_batch[:0]

                local_batch_size = x_batch.shape[0]

                sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)

                if sync_model:
                    sync_epoch = True

                if model_sync_count == 0 and not self.initial_model_sync:
                    sync_model = False

                model_sync_count += 1

                if i_batch < val_batches_min:
                    rank_mask = [1] * self.comm_size
                else:
                    rank_mask = self.comm.allgather(min(1, local_batch_size)) if self.comm else [min(1, local_batch_size)]
                rank_avail = sum(rank_mask)

                if rank_avail <= 0:
                    break

                if rank_avail < self.model_sync_min_avail:
                    sync_model = False

                val_batch_loss = self._evaluate_batch(x_batch, y_batch, sync_model=False and sync_model)

                if batch_size <= 0:
                    continue

                val_total_loss, val_batch_count, string = \
                    self._update_running_average(val_batch_loss, val_total_loss,
                                                 val_batch_count, batch_size, prefix="val_")
                if self.comm_rank == 0:
                    pbar.set_postfix_str(s=f"{train_string}, {string}", refresh=True)

            if self.comm_rank == 0:
                for c in range(len(self.loss_and_metrics)):
                    self.history["val_" + self.loss_and_metrics[c]].append(val_total_loss[c])

            for lr_sched in self.lr_schedulers:
                lr_sched.on_epoch_end(train_total_loss, val_total_loss)
                if lr_sched.stop_training:
                    terminate = True

            if self.comm_rank == 0:
                pbar.close()
                # Sleep for half a second to allow pbar to write its output before returning
                time.sleep(.5)

            if sync_epoch:
                global_terminate = self.comm.allreduce(terminate, op=MPI.LAND) if self.comm else terminate

            if global_terminate:
                break

        # Syncronize model
        if self.final_model_sync:
            self._weight_update(gradient=False, blocking=self.blocking_mpi)

        self.tracer.define_event_types(self)
        return self.history
    # --- END train_dataset --- #

    def _train_batch[T:Array](self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model=True) -> T:
        self.mode = ModelModeEnum.TRAIN

        # LR schedulers begin
        for lr_sched in self.lr_schedulers:
            lr_sched.on_batch_begin()

        x, y_targ = self._sync_x_y(x_batch, y_batch)

        last_layer = len(self.layers) - 1

        has_batch = x_batch.shape[0] > 0

        if has_batch:
            # Forward pass (FP)
            for i in range(len(self.layers)):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            loss, dx = self.loss_func(x, y_targ, self.num_real_batches)
        else:
            if y_targ.shape[0] != x_batch.shape[0]:
                raise ValueError(f"y_targ.shape[0] ({y_targ.shape[0]}) and x_batch.shape[0] ({x_batch.shape[0]}) must have the same value.")
            loss, dx = 0.0, y_targ

        self.total_metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, comm=sync_model)

        if has_batch:
            # Backward pass (BP)
            for i in range(last_layer, -1, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx = self.layers[i].backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        if self.enable_cudnn:
            self.stream.synchronize()

        # Gradient update
        if sync_model:
            self._weight_update(gradient=True, blocking=self.blocking_mpi)

        if has_batch or sync_model:

            # Optimizer
            for i in range(last_layer, -1, -1):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.UPDATE_DW)
                self.layers[i].update_weights(self.optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        # Weight update
        if self.model_sync_freq > 0 and sync_model:
            self._weight_update(gradient=False, blocking=self.blocking_mpi)

        if self.enable_cudnn:
            for i in range(last_layer, -1, -1):
                if self.layers[i].grad_vars:
                    self.layers[i].stream_2.synchronize()

        # LR schedulers end
        for lr_sched in self.lr_schedulers:
            lr_sched.on_batch_end(self)

        return self.total_metrics
    # --- END _train_batch --- #

    def _compute_rank_weight(self, mask: list[int], part: DatasetEnum) -> float:
        # TODO Move "all" and "avail2all" to an Enum?
        match self.model_sync_participation:
            case "all":
                comm_nsamples = self.comm_nsamples[part]
            case "avail2all":
                if mask[self.comm_rank]:
                    comm_nsamples = [nsamples for nsamples, mask in zip(self.comm_nsamples[part], mask) if mask]
                else:
                    return 0.0
            case _:
                raise SystemExit(f"Model synchronization participation option '{self.model_sync_participation}' not recognized.")

        min_nsamples, max_nsamples, total_nsamples = min(comm_nsamples), max(comm_nsamples), sum(comm_nsamples)
        comm_size = len(comm_nsamples)

        # TODO Move "avg", "wavg", "invwavg" to an Enum?
        match self.model_sync_alg:
            case "avg":
                return 1.0 / comm_size
            case "wavg":
                return self.dataset._nsamples[part] / total_nsamples
            case "invwavg":
                inverse_nsamples = min_nsamples + (max_nsamples - self.dataset._nsamples[part])
                return inverse_nsamples / total_nsamples
            case _:
                raise SystemExit(f"Model synchronization algorithm option '{self.model_sync_alg}' not recognized.")
    # --- END _compute_rank_weight --- #

    def _evaluate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model=True) -> np.ndarray:
        self.mode = ModelModeEnum.EVALUATE

        x, y_targ = self._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        # Forward pass (FP)
        if has_batch:
            for i in range(len(self.layers)):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            y_pred = self.layers[-1].y
            loss, _ = self.loss_func(y_pred, y_targ, self.num_real_batches)
        else:
            y_pred = self.layers[-1].y
            loss = 0.0

        self.total_metrics, _ = self._compute_metrics_funcs(y_pred, y_targ, loss, comm=sync_model)

        return self.total_metrics
    # --- END _evaluate_batch --- #

    @ensure_model_is_runnable
    def evaluate_dataset(self, bar_width=BAR_WIDTH):
        if self.enable_cudnn and self.y_batch is None:
            self.y_batch = pydtnn.backends.gpu.tensor_gpu.TensorGPU(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)

        self.comm_nsamples = list(zip(*self.comm.allgather(self.dataset._nsamples) if self.comm else [self.dataset._nsamples]))

        test_batches_min = min(self.comm_nsamples[DatasetEnum.TEST]) / (self.batch_size * self.nprocs)

        test_batch_generator = self.dataset.get_test_generator()

        if self.comm_rank == 0:
            test_total_loss, test_batch_count = np.zeros(len(self.loss_and_metrics)), 0
            pbar = tqdm(total=self.dataset.test_nsamples, ncols=bar_width,
                        ascii=" ▁▂▃▄▅▆▇█", smoothing=0.3,
                        desc="Testing", unit=" samples")

        model_sync_count = 0
        for i_batch, (x_batch, y_batch, batch_size) in enumerate(test_batch_generator):
            local_batch_size = x_batch.shape[0]

            sync_model = (self.model_sync_freq <= 0) or (model_sync_count % self.model_sync_freq == 0)

            if model_sync_count == 0 and not self.initial_model_sync:
                sync_model = False

            model_sync_count += 1

            if i_batch < test_batches_min:
                rank_mask = [1] * self.comm_size
            else:
                rank_mask = self.comm.allgather(min(1, local_batch_size)) if self.comm else [min(1, local_batch_size)]
            rank_avail = sum(rank_mask)

            if rank_avail <= 0:
                break

            if rank_avail < self.model_sync_min_avail:
                sync_model = False

            tic = timer()
            test_batch_loss = self._evaluate_batch(x_batch, y_batch, sync_model=sync_model)
            toc = timer()

            if batch_size <= 0:
                continue

            if self.comm_rank == 0:
                # noinspection PyUnboundLocalVariable
                test_total_loss, test_batch_count, string = \
                    self._update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size,
                                                 prefix="test_")
                # noinspection PyUnboundLocalVariable
                pbar.set_postfix_str(s=string, refresh=True)
                pbar.update(batch_size)
                self.perf_counter.add_testing_time_and_batch_size(self._evaluate_round, toc - tic, batch_size)

        # Increment self._evaluate_round
        self._evaluate_round += 1

        if self.comm_rank == 0:
            pbar.close()
            # Sleep for half a second to allow pbar to write its output before returning
            time.sleep(.5)
