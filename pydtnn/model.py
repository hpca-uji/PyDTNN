"""
PyDTNN model
"""
from pydtnn.utils.memory_pool import PrivateMemory, PreallocMemory
from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import Array, NetworkAlgEnum, ArrayShape, Parameters
from pydtnn.utils.tensor import SampleFormat, TensorFormat, format_reshape, encode_shape, encode_tensor, decode_shape, decode_tensor
from pydtnn.utils.performance_counter import PerformanceCounter
from pydtnn.tracers.tracer import Tracer
from pydtnn.tracers.simple_tracer_pmlib import SimpleTracerPMLib
from pydtnn.tracers.simple_tracer_gpu import SimpleTracerPycuda
from pydtnn.tracers.simple_tracer import SimpleTracer
from pydtnn.tracers.extrae_tracer import ExtraeTracer
from pydtnn.tracers.events import PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.performance_models import allreduce_time
from pydtnn.parser import PydtnnArgumentParser
from pydtnn.backends.fuse.layers.layer import select as select_fuse_layer
from pydtnn.schedulers.scheduler import select as select_scheduler
from pydtnn.models.model import select as select_model
from pydtnn.metrics.metric import select as select_metric
from pydtnn.optimizers.optimizer import select as select_optimizer
from pydtnn.losses.loss import select as select_loss
from pydtnn.datasets.dataset import select as select_dataset
from pydtnn.losses.loss import Loss
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.backends.fuse.layers.layer import LayerFuse as FusedLayerMixIn
from pydtnn.abstract.layerable import Layerable
from pydtnn.datasets.dataset import Dataset
from pydtnn.libs.mpi.rc import proto as PROTOCOL
from pydtnn.activations.relu import Relu
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn import utils
from pydtnn import hostname, ranks_per_node, num_gpus, nccl_comm, cudnn_handle, cublas_handle, context, stream
from pydtnn import MPI_MODULE, Cudnn_Handle_Type, Cublas_Handle_Type, MPI, drv, gpuarray, nccl, cudnn, cublas  # type: ignore (cublas exist)
from tqdm import tqdm
import numpy as np
from collections import abc
from warnings import warn
from typing import TYPE_CHECKING, Any, Literal
from types import ModuleType
from timeit import default_timer as timer
from functools import reduce
import time
import operator
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


def get_tracer(tracer_output: str, tracing: bool, comm: MPI_COMM | None, enable_cudnn: bool,
               tracer_pmlib_server: str, tracer_pmlib_port: int, tracer_pmlib_device: str) -> Tracer:

    if tracer_output == "":
        tracer = ExtraeTracer(tracing)
    else:
        if enable_cudnn:
            tracer = SimpleTracerPycuda(tracing, tracer_output, comm)
        else:
            if tracer_pmlib_device != "":
                tracer = SimpleTracerPMLib(tracing, tracer_output, comm, tracer_pmlib_server, tracer_pmlib_port, tracer_pmlib_device)
            else:
                tracer = SimpleTracer(tracing, tracer_output, comm)
    return tracer


def get_tensor_format(tensor_format: Literal["AUTO", "NCHW", "NHWC"] = "AUTO", gpu: bool = False) -> TensorFormat:
    match tensor_format.upper():
        case "AUTO":
            return TensorFormat.NCHW if gpu else TensorFormat.NHWC
        case "NCHW":
            return TensorFormat.NCHW
        case "NHWC":
            return TensorFormat.NHWC
        case _:
            raise NotImplementedError(f"\'{tensor_format}\' is not supported.")


def get_batch_size(local_size: int | None, global_size: int | None, comm_size: int, default: int = DEFAULT_BACH_SIZE) -> int:
    if local_size and global_size:
        raise ValueError("Can not define 'local_batch_size' and 'global_batch_size' simultaneously")

    if global_size:
        # NOTE: Using comm_size instead of nprocs might not be appropriate,
        #       as it differs to how global_batch_size is defined elsewhere,
        #       but for now it just a parser option difference that helps testing
        batch_size = global_size // comm_size
    elif local_size:
        batch_size = local_size
    else:
        batch_size = default

    if batch_size < 1:
        raise ValueError(f"'batch_size' ({batch_size}) too small or too many processes (num processes: {comm_size})")

    return batch_size


class CudnnDataType(enum.StrEnum):
    FLAOT64 = "CUDNN_DATA_DOUBLE"
    FLOAT32 = "CUDNN_DATA_FLOAT"
    INT8 = "CUDNN_DATA_INT8"
    INT32 = "CUDNN_DATA_INT32"


class Model[T: Array]:
    """
    PyDTNN Model
    """

    class Mode(enum.StrEnum):
        EVALUATE = enum.auto()
        TRAIN = enum.auto()
        UNSPECIFIED = enum.auto()

    class SyncAlgorithm(enum.StrEnum):
        AVG = enum.auto()
        WAVG = enum.auto()
        INVAVG = enum.auto()

    class SyncParticipation(enum.StrEnum):
        ALL = enum.auto()
        AVAIL2ALL = enum.auto()

# Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
#   NOTE: The following parameters come from "Parser"
    steps_per_epoch: int
    cpu_speed: float
    memory_bw: float
    network_bw: float
    network_lat: float
    network_alg: NetworkAlgEnum
    loss_func_name: str
    num_epochs: int
    model_sync_freq: int
    final_model_sync: bool
    test_as_validation: bool
    validation_split: float
    use_synthetic_data: bool
    dataset_train_path: str
    dataset_test_path: str
    evaluate_only: bool
    evaluate_on_train: bool
    profile: bool
    history_file: str
    model_sync_min_avail: int
    dataset_name: str
    shared_storage: bool
    encryption_name: str
    augment_flip: float
    augment_crop_size: int
    augment_crop: float
    transform_crop: bool
    transform_crop_perc: int
    transform_resize: bool
    transform_resize_dsize: int
    initial_model_sync: bool
    dataset_percentage: float
    use_mpi_buffers: bool
    # enable_memory_cache: bool
    gpus_per_node: int
    weights_and_bias_filename: str
    learning_rate_scaling: bool
    metrics: str
    use_memory_pool: bool
    augment_shuffle: bool
    normalize: bool
    transform_resize_size: int
    normalize_offset: float
    normalize_scale: float
    model_sync_quantize: bool
    model_sync_dtype: np.dtype
# ------------

    rank_weight: float
    comm_rank: int
    comm_size: int
    rank: int
    nprocs: int
    learning_rate: float
    MPI: MPI_MODULE | None
    comm: MPI_COMM | None

    nccl_type: Any | None
    nccl_comm: Any | None
    cudnn_handle: Cudnn_Handle_Type | None
    cublas_handle: Cublas_Handle_Type | None
    stream: Any  # drv.Stream
    cudnn_dtype: int
    input_shape: ArrayShape
    output_shape: ArrayShape

    dtype: np.dtype

    real_batch_size: int
    nparams: int

    y_batch: T
    history: dict[str, list[np.ndarray]]
    loss_func: Loss
    metrics_funcs: list[Metric]
    loss_and_metrics: list[str]
    total_metrics: np.ndarray

    cuda_grid: tuple[int, int, int]
    cuda_block: tuple[int, int, int]

    def __init__(self, **kwargs):

        # Get default values from parser and update them from the received kwargs
        self.kwargs: dict[str, Any] = PydtnnArgumentParser().get_default_values()
        self.kwargs.update(kwargs)

        # Attributes related to the given arguments
        self.blocking_mpi: bool = self.use_blocking_mpi
        self.enable_cudnn = gpuarray is not None and drv is not None and cublas is not None
        self.gpudirect: bool = self.enable_gpudirect
        self.enable_nccl: bool = self.enable_nccl
        self.dtype: np.dtype = np.dtype(self.dtype)
        self.memory: PrivateMemory = None  # type: ignore (it will be intialized later if "self.use_memory_pool" is True)

        self.nparams = 0
        self.memory_used = 0
        self.tmp_memory_used = 0

        # Set MPI and comm
        self._mpi_init()

        # Set performance counter
        self.perf_counter = PerformanceCounter()

        # Layers' attributes
        self.layers: list[Layerable] = []
        self.layer_id_generator: abc.Iterator[int] = iter(itertools.count())

        # Set current mode to unspecified
        self.mode: Model.Mode = Model.Mode.UNSPECIFIED

        # Memory cache optimization
        # if self.enable_memory_cache:
        #     MemoryCache.enable()
        # else:
        #     MemoryCache.disable()

        self.memory_cls = PreallocMemory if self.shared_tmp_memory else PrivateMemory

        # Set tracer
        self.tracer = get_tracer(tracer_output=self.tracer_output, tracing=self.tracing, comm=self.comm, enable_cudnn=self.enable_cudnn,
                                 tracer_pmlib_server=self.tracer_pmlib_server, tracer_pmlib_port=self.tracer_pmlib_port,
                                 tracer_pmlib_device=self.tracer_pmlib_device)

        # Cuda
        if self.enable_cudnn:
            self._cudnn_init()

        # Data format
        self.tensor_format: TensorFormat = get_tensor_format(tensor_format=self.tensor_format, gpu=self.enable_cudnn)  # type: ignore

        # Disable BestOf globally if not enabled
        # if self.enable_best_of is False:
        #     BestOf.use_always_the_first_alternative()

        self.batch_size = get_batch_size(local_size=self.batch_size, global_size=self.global_batch_size, comm_size=self.comm_size)

        # Attributes that will be properly initialized elsewhere

        # ---

        # Encryption
        if self.encryption_name:
            self.crypt = self._crypt_init(self.encryption_name)

        else:
            self.crypt = None

        # Load weights and bias
        if self.weights_and_bias_filename:
            self.load_weights_and_bias(self.weights_and_bias_filename)
        # Dataset
        if self.dataset_name:
            self.dataset: Dataset = select_dataset(self.dataset_name)(self)

        # Optimizers and LRSchedulers
        if self.learning_rate_scaling:
            # using comm_size instead of nprocs might not be appropriate,
            # as it differs to how learning_rate is defined elsewhere,
            # but for now it just a parser option difference that helps testing
            self.learning_rate = self.learning_rate / self.comm_size

        self.optimizer = select_optimizer(self.optimizer_name).from_model(self)
        self.optimizer._init_backend_with_model(self)

        self.schedulers = [
            select_scheduler(scheduler_name).from_model(self)
            for scheduler_name in filter(None, self.schedulers_names.split(","))
        ]
        for scheduler in self.schedulers:
            scheduler.model = self

        # Metrics list
        self.metrics_list: list[str] = [m for m in self.metrics.replace(" ", "").split(",")]

        # Private attributes
        self._evaluate_round: int = 0
        self._is_model_init: bool = False

        # Read the model (must be the last action, as it calls self._model_init() if there is a model)
        self.model_name: str | None = self.kwargs.get("model_name")
        if self.model_name:
            self._read_model(self.model_name)

        # Synchronization parameters
        # NOTE: This parameter come from Parser.
        self.model_sync_algo = Model.SyncAlgorithm(self.model_sync_algo)

        # NOTE: This parameter come from Parser.
        self.model_sync_participation = Model.SyncParticipation(self.kwargs["model_sync_participation"])

    def _mpi_init(self) -> None:
        # Communication type
        if self.parallel_data or self.parallel_pipeline:
            if not MPI:
                raise ValueError("Please, install mpi4py to allow parallel MPI execution!")
            self.MPI, self.comm = (MPI, MPI.COMM_WORLD)
        else:
            self.MPI, self.comm = (None, None)

        # Communication size
        self.rank_weight = 1.0
        self.comm_rank = self.rank = 0
        self.comm_size = self.nprocs = 1
        if self.comm:
            self.comm_rank = self.comm.Get_rank()
            self.comm_size = self.comm.Get_size()
            if self.shared_storage:
                self.rank = self.comm_rank
                self.nprocs = self.comm_size

        # Communication method
        match self.use_mpi_buffers:
            case None:
                self.use_mpi_buffers = PROTOCOL is None
            case bool():
                pass
            case _:
                raise ValueError(f"MPI buffers option '{self.use_mpi_buffers}' not recognized.")

    def _cudnn_init(self) -> None:
        LIMIT_THREADS_AND_BLOCKS = 1024
        self.cuda_threads = min(self.batch_size, LIMIT_THREADS_AND_BLOCKS)
        self.cuda_blocks = (max(self.batch_size, LIMIT_THREADS_AND_BLOCKS) // self.cuda_threads) + 1
        # NOTE: Seems that in PyDTNN, usually the ".x" (blockIdx.x, threadIdx.x, ...) is the only dimension used.
        self.cuda_grid = (self.cuda_blocks, 1, 1)
        self.cuda_block = (self.cuda_threads, 1, 1)

        assert drv is not None
        assert context is not None
        assert cudnn_handle is not None
        assert cublas_handle is not None
        assert stream is not None

        if not self.gpudirect and self.enable_nccl:
            raise RuntimeError("It is necessary to have gpudirect active to work with NCCL.")

        if self.comm and self.enable_nccl:
            assert nccl is not None
            assert nccl_comm is not None

            nccl_types = {np.float64: nccl.DataType.Float64,
                          np.float32: nccl.DataType.Float32,
                          np.int8: nccl.DataType.Int8,
                          np.int32: nccl.DataType.Int32}

            nccl_type = nccl_types.get(self.dtype, nccl.DataType.Float32)

            if ranks_per_node[hostname] > num_gpus:
                raise ValueError("Not able to run more processes than GPUs per node!")
        else:
            nccl_type = None

        self.tracer.set_stream(stream)

        cudnn_types = {np.float64: CudnnDataType.FLAOT64,
                       np.float32: CudnnDataType.FLOAT32,
                       np.int8: CudnnDataType.INT8,
                       np.int32: CudnnDataType.INT32}

        cudnn_type: str = cudnn_types.get(self.dtype, CudnnDataType.FLOAT32)
        cudnn_dtype: int = cudnn.cudnnDataType[cudnn_type]

        self.nccl_type = nccl_type
        self.nccl_comm = nccl_comm
        self.cudnn_handle = cudnn_handle
        self.cublas_handle = cublas_handle
        self.stream = stream
        self.cudnn_dtype = cudnn_dtype

    def _ensure_model_runable(self) -> None:
        if not self.layers:
            warn_text = "The model has no layers in it."
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)
        elif not self.dataset:
            raise ValueError("There is no dataset and the model has layers.")
        self._model_init()

    @property
    def dataset_path(self) -> str:
        """Raw dataset path with rank substituted"""
        return utils.string_substitute(self.kwargs["dataset_path"], rank=self.comm_rank)

    def __getattr__(self, item) -> Any:
        return self.kwargs.get(item)

    def _crypt_init(self, encryption_name: str) -> "polyhe.Context":
        """Initialize encryption context"""
        if polyhe is None:
            raise RuntimeError("uHE is not avaliable, but is requiested!")

        backend = polyhe.Backend(encryption_name)
        options = polyhe.Options(
            slots=self.encryption_slots,
            scale=self.encryption_scale,
            security=self.encryption_security
        )

        if self.comm_rank == 0:
            crypt = polyhe.new(backend, options)

        if self.comm:
            crypt = self.comm.bcast(crypt if self.comm_rank == 0 else None)

        assert crypt is not None
        if self.enable_nccl:
            warn_text = "If NCCL is active, encryption is disabled"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)

        return crypt

    def _read_model(self, model_name: str) -> None:
        create_model = select_model(model_name)

        # NOTE: Dataset is always in NCHW
        # Change input_shape to model.tensor_format
        input_shape = format_reshape(self.dataset.input_shape, SampleFormat.CHW, self.tensor_format.as_sample())  # type: ignore
        if len(input_shape) != 3:
            warn_text = f"Input layer does not have 3 dimensions ({input_shape}), it may cause issues!"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)
        launch_shape_warning = len(input_shape) == 3 and not (input_shape[0] > input_shape[2]) if self.tensor_format is TensorFormat.NHWC \
            else len(input_shape) == 3 and not (input_shape[0] < input_shape[1])
        if launch_shape_warning:
            warn_text = f"Input layer shape {input_shape} may not be in {self.tensor_format} format, regardless of model format! "
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)
        output_shape = tuple(self.dataset.output_shape)

        layers = create_model(input_shape, output_shape)
        self.add_layers(layers)  # type: ignore

    def encode_shape(self, shape: ArrayShape) -> ArrayShape:
        """Transform the shape from `NCHW` order to `model.tensor_format` order (supports 4 or 3 dimensions)"""
        return encode_shape(shape, self.tensor_format)

    def decode_shape(self, shape: ArrayShape) -> ArrayShape:
        """Transform the shape from `model.tensor_format` order to `NCHW` order (supports 4 or 3 dimensions)."""
        return decode_shape(shape, self.tensor_format)

    def encode_tensor(self, data: np.ndarray) -> np.ndarray:
        """Transpose elements of data from `NCHW` format to `model.tensor_format` format (supports 4 or 3 dimensions)."""
        return encode_tensor(data, self.tensor_format)  # type: ignore (TensorGPU does not have transpose yet)

    def decode_tensor(self, data: np.ndarray) -> np.ndarray:
        """Transpose elements of data from `model.tensor_format` format to `NCHW` format (supports 4 or 3 dimensions)."""
        return decode_tensor(data, self.tensor_format)  # type: ignore (TensorGPU does not have transpose yet)

    def _layer_reduce_encode(self, data: np.ndarray):
        data *= self.rank_weight

        if self.model_sync_quantize:
            data = np.astype(data, self.model_sync_dtype)

        if self.crypt:
            data = self.crypt.encrypt(data)  # type: ignore

        return data

    def _layer_reduce_decode(self, data) -> np.ndarray:

        if self.crypt:
            data = self.crypt.decrypt(data)

        if self.model_sync_quantize:
            data = np.astype(data, self.dtype)

        return data

    def _layer_reduce_sync(self, data: np.ndarray) -> np.ndarray:
        assert self.comm is not None, "Reduce without communicator"
        if self.use_mpi_buffers:
            self.comm.Allreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        else:
            data = self.comm.allreduce(data, op=MPI.SUM)
        return data

    def _layer_reduce_async(self, data):
        assert self.comm is not None, "Reduce without communicator"
        if self.use_mpi_buffers:
            req = self.comm.Iallreduce(MPI.IN_PLACE, data, op=MPI.SUM)
        else:
            req = self.comm.iallreduce(data, op=MPI.SUM)
        return req

    def _layer_reduce_wait(self, data, request):
        if (response := request.wait()) is not None:
            data = response
        return data

    def _show_props(self) -> dict:
        props = {}

        if self.model_name:
            props["name"] = self.model_name

        if self.dataset_name:
            props["dataset"] = self.dataset_name

        if self.nparams > 0:
            props["params"] = self.nparams

        if self.memory_used > 0:
            memory = utils.convert_size_bytes(self.memory_used)
            if self.tmp_memory_used > 0:
                tmp_memory = utils.convert_size_bytes(self.tmp_memory_used)
                memory = f"{memory} ({tmp_memory} tmp)"
            props["memory"] = memory

        if self.optimizer:
            optimizer_memory = utils.convert_size_bytes(self.optimizer.memory_used)
            if self.optimizer.tmp_memory_used > 0:
                optimizer_tmp_memory = utils.convert_size_bytes(self.optimizer.tmp_memory_used)
                optimizer_memory = f"{optimizer_memory} ({optimizer_tmp_memory} tmp)"
            props["optimizer-memory"] = optimizer_memory

        if self.loss_func:
            loss_memory = utils.convert_size_bytes(self.loss_func.memory_used)
            if self.loss_func.tmp_memory_used > 0:
                loss_tmp_memory = utils.convert_size_bytes(self.loss_func.tmp_memory_used)
                loss_memory = f"{loss_memory} ({loss_tmp_memory} tmp)"
            props["loss-memory"] = loss_memory

        if self.metrics_funcs:
            metrics_size = 0
            metric_temp_size = 0
            for metric in self.metrics_funcs:
                metrics_size += metric.memory_used
                metric_temp_size += metric.tmp_memory_used
            metrics_memory = utils.convert_size_bytes(metrics_size)
            if metric_temp_size > 0:
                metrics_tmp_memory = utils.convert_size_bytes(metric_temp_size)
                metrics_memory = f"{metrics_memory} ({metrics_tmp_memory} tmp)"
            props["metrics-memory"] = metrics_memory

        if self.layers:
            props["input"] = self.layers[0].shape
            props["output"] = self.layers[-1].shape
            props["batch-size"] = self.batch_size
            props["layers"] = len(self.get_all_layers())

        return props

    def __repr__(self) -> str:
        props = " ".join(
            f"{key}={value!r}"
            for key, value in self._show_props().items()
        )

        return f"<{self.__class__.__name__} {props}>"

    def show_layers(self) -> None:
        struct: dict[str, int] = {}
        all_props = {
            layer.id: layer._show_props()
            for layer in self.get_all_layers()
        }

        # Calculate headers and sizes
        for props in sorted(all_props.values(), key=lambda props: (-len(props), *props)):
            for key, value in props.items():
                struct[key] = max(struct.get(key, len(key)), len(str(value)))

        # Add header padding
        for header, size in struct.items():
            struct[header] += 2

        # Generate separator
        sep = ""
        for header, size in struct.items():
            sep += "+" + "-" * size
        sep += "+"

        # Show header
        _show = [""]
        _show.append(sep)
        _show.append("")
        for header, size in struct.items():
            _show[-1] += (f"|{header.replace('-', ' ').capitalize():^{size}s}")
        _show[-1] += ("|")

        # Show layers
        top_layers = {layer.id for layer in self.layers}
        for layer_id, props in all_props.items():
            if layer_id in top_layers:
                _show.append(sep)
            _show.append("")
            for header, size in struct.items():
                value = props.get(header, "")
                _show[-1] += (f"|{str(value):^{size}s}")
            _show[-1] += ("|")
        _show.append(sep)
        logger.info('\n'.join(_show))

    def show_model(self) -> None:
        key: str = "Model Summary"
        _show = [""]
        _show.append(key)
        _show.append("=" * len(key))
        for key, value in self._show_props().items():
            _show.append(f"- {key.replace('-', ' ').capitalize()}: {value}")
        logger.info('\n'.join(_show))

    def show(self) -> None:
        self.show_model()
        self.show_layers()

    def print_in_convdirect_format(self) -> None:
        line = "#l\tkn\two\tho\tt\tkh\tkw\tci\twi\thi"
        logger.info(line)
        for layer in self.layers:
            layer.print_in_convdirect_format()

    def add(self, layer: Layerable[T]) -> None:
        layer._init_backend_with_model(self)

        if self.layers:
            prev_shape = self.layers[-1].shape
            y = self.layers[-1].y
        else:
            prev_shape = ()
            y = None

        layer._model_init(prev_shape, y)

        self.nparams += layer.nparams
        # self.memory_used += layer.memory_used
        # self.tmp_memory_used += layer.tmp_memory_used
        self.layers.append(layer)

        if layer.act:
            self.add(layer.act())

    def add_layers(self, list_layers: list[Layerable[T]]) -> None:
        for layer in list_layers:
            self.add(layer)
    # --- END add_layers ---

    def get_all_layers(self, from_layers: list[Layerable[T]] | None = None) -> list[Layerable[T]]:
        if from_layers is None:
            from_layers = self.layers
        this_recursion_layers = []
        for layer in from_layers:
            this_recursion_layers.append(layer)
            children = layer.children
            this_recursion_layers += self.get_all_layers(children)
        return this_recursion_layers

    def _select_fusion_3(self, fused_layers: list) -> tuple[str | None, list[Layerable | FusedLayerMixIn | None]]:
        layer2 = fused_layers[-1] if len(fused_layers) > 0 else None
        layer1 = fused_layers[-2] if len(fused_layers) > 1 else None
        layer0 = fused_layers[-3] if len(fused_layers) > 2 else None

        layer_name = None

        match (layer0, layer1, layer2):
            case (_, FusedLayerMixIn(), _): pass  # else: layer_name = None
            case (Conv2D(), BatchNormalization(), Relu()):
                if self.enable_fused_conv_bn_relu:
                    layer_name = "conv_2d_batch_normalization_relu"
                # else: layer_name = None
            case _: pass  # else: layer_name = None

        return layer_name, [layer0, layer1, layer2]
    # ----

    def _select_fusion_2(self, fused_layers: list) -> tuple[str | None, list[Layerable | FusedLayerMixIn | None]]:
        layer2 = fused_layers[-1] if len(fused_layers) > 0 else None
        layer1 = fused_layers[-2] if len(fused_layers) > 1 else None

        layer_name = None

        match (layer1, layer2):
            case (FusedLayerMixIn(), _): pass  # else: layer_name = None
            # else: layer_name = None
            case (Conv2D(), BatchNormalization()):
                if self.enable_fused_conv_bn:
                    layer_name = "conv_2d_batch_normalization"
                # else: layer_name = None
            case (Conv2D(), Relu()):
                if self.enable_fused_conv_relu:
                    layer_name = "conv_2d_relu"
                # else: layer_name = None
            case (BatchNormalization(), Relu()):
                if self.enable_fused_bn_relu:
                    layer_name = "batch_normalization_relu"
                # else: layer_name = None
            case _: pass  # else: layer_name = None

        return layer_name, [layer1, layer2]
    # ----

    def _layer_fusion(self, layers: list[Layerable], switch_fusion: abc.Callable) -> None:
        i = 0
        while i < len(layers):
            curr_layer = layers[i]

            # Recurse if layer group
            for j, p in enumerate(curr_layer.paths):
                self._layer_fusion(curr_layer.paths[j], switch_fusion)

            layer_name, layers_to_fuse = switch_fusion(layers[:i])

            if layer_name:
                dict_params = reduce(operator.or_, (layer.__dict__ for layer in reversed(layers_to_fuse)))
                logger.info(f"Fusing {' + '.join(map(lambda layer: layer.name_with_id, layers_to_fuse))}")
                fused_layer = select_fuse_layer(layer_name)

                new_curr_layer = fused_layer(from_parent=dict_params)  # type: ignore (it's okay)
                new_curr_layer._init_backend_with_model(self)
                new_curr_layer.__dict__.update(dict_params)
                try:
                    new_curr_layer._model_init(prev_shape=layers_to_fuse[0].prev_shape, x=layers_to_fuse[0].x)
                except Exception as e:
                    warn_text = f"Aborted fusion, {e}"
                    logger.warning(warn_text)
                    warn(warn_text, RuntimeWarning)
                else:
                    start = i - len(layers_to_fuse)
                    del layers[start: i]
                    layers.insert(start, new_curr_layer)
                    i -= len(layers_to_fuse)
            i += 1

    def _apply_layer_fusion(self):
        """ Apply layer fusion in a recursive manner """

        if not self.enable_cudnn and any([self.enable_fused_bn_relu, self.enable_fused_conv_relu, self.enable_fused_conv_bn, self.enable_fused_conv_bn_relu]):
            # NOTE: 1st the 3 layers fusion, then the rest:
            self.backend = f"layers:fuse;{self.backend}"
            self._layer_fusion(self.layers, self._select_fusion_3)
            self._layer_fusion(self.layers, self._select_fusion_2)

    def _model_init(self):
        if self._is_model_init:
            return
        self._is_model_init = True

        self._apply_layer_fusion()

        temp_memory_size = []
        self._output_shape = (self.batch_size, *self.layers[-1].shape)

        self.loss_func = select_loss(self.loss_func_name)()
        self.loss_func._init_backend_with_model(self)
        self.loss_func._model_init()
        self.memory_used += self.loss_func.memory_used
        temp_memory_size.append(self.loss_func.tmp_memory_used)

        self.metrics_funcs = [select_metric(m)() for m in self.metrics_list]
        self.metrics_funcs.sort(key=lambda metric: metric.order)

        for metric in self.metrics_funcs:
            metric._init_backend_with_model(self)
            metric._model_init()
            self.memory_used += metric.memory_used
            temp_memory_size.append(metric.tmp_memory_used)

        self.loss_and_metrics = [self.loss_func_name] + self.metrics_list
        self.loss_and_metrics_format = [self.loss_func.format] + [metric.format for metric in self.metrics_funcs]
        self.total_metrics = np.array([0] + [0 for func in self.metrics_funcs], dtype=self.dtype)
        self.tracer.define_event_types(self)

        self.optimizer._model_init(self.get_all_layers(self.layers))
        self.memory_used += self.optimizer.memory_used
        temp_memory_size.append(self.optimizer.tmp_memory_used)

        for layer in self.layers:
            self.memory_used += layer.memory_used
            temp_memory_size.append(layer.tmp_memory_used)

        self.tmp_memory_used = self.memory_cls._total(*temp_memory_size)
        self.memory_used += self.tmp_memory_used
        self.memory = self.memory_cls(size=self.tmp_memory_used)

        for layer in self.get_all_layers():
            layer._post_init()

        for metric in self.metrics_funcs:
            metric._post_init()

        self.loss_func._post_init()
        self.optimizer._post_init()

        # ----

    def export(self) -> dict[str, Any]:
        data = {}

        if self.model_name is not None:
            data[Parameters.MODEL_NAME] = self.model_name

        data[Parameters.LAYERS] = [
            layer.export()
            for layer in self.layers
        ]

        return data

    def import_(self, data: "dict[str, Any] | Model") -> None:
        if isinstance(data, Model):
            data = data.export()

        model_name = str(data.get(Parameters.MODEL_NAME))
        if model_name != self.model_name:
            warn_text = f"Importing from different models! (self: {self.model_name}, got: {model_name})"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)

        for layer, data in zip(self.layers, data[Parameters.LAYERS]):
            layer.import_(data)  # type: ignore (It is the right data type.)

    def load_weights_and_bias(self, filename: str) -> None:
        """
        ARGS:
            filename: Path to the file with the weights and biases to load.
        """
        with np.load(filename, allow_pickle=True) as data:
            self.import_(data)

    def store_weights_and_bias(self, filename: str, compress=True) -> None:
        """
        ARGS:
            filename: Path to the file were the weights and biases will be stored.
        """
        save = np.savez_compressed if compress else np.savez
        save(filename, **self.export())

    def calculate_time(self) -> np.ndarray:
        # Total elapsed_time, Comp elapsed_time, Memo elapsed_time, Net elapsed_time
        total_time: np.ndarray = np.zeros((4,), dtype=np.float32)

        # Forward pass (FP)
        for layer in self.layers:
            total_time += layer.fwd_time

        if self.blocking_mpi:
            # Blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in self.layers:
                total_time += layer.bwd_time

            # Weight update (WU)
            for layer in self.layers:
                weights_size = 0 if (weights := layer.weights) is None else weights.size
                biases_size = 0 if (biases := layer.biases) is None else biases.size
                if self.comm and weights_size > 0:
                    total_time += allreduce_time(weights_size + biases_size,
                                                 self.cpu_speed, self.network_bw, self.network_lat,
                                                 self.network_alg, self.nprocs, self.dtype)
        else:
            total_time_iar: int = 0
            # Non-blocking MPI
            # Back propagation. Gradient computation (GC) and weights update (WU)
            for layer in self.layers:
                total_time += layer.bwd_time
                weights_size = 0 if (weights := layer.weights) is None else weights.size
                biases_size = 0 if (biases := layer.biases) is None else biases.size
                if self.comm and weights_size > 0:
                    time_iar = allreduce_time(weights_size + biases_size,
                                              self.cpu_speed, self.network_bw, self.network_lat,
                                              self.network_alg, self.nprocs, self.dtype)
                    total_time[3] += time_iar[3]
                    total_time_iar = max(total_time[0], total_time_iar) + time_iar[0]

            total_time[0] = max(total_time[0], total_time_iar)

        return total_time

    def _compute_metrics_funcs(self, y_pred: T, y_targ: T, loss: float, blocking=True, comm=True) -> tuple[np.ndarray, None] | tuple[None, Any]:
        loss_req: Any | None = None
        _losses: np.ndarray | None

        if y_targ.shape[0] > 0:
            metrics = [func.compute(y_pred, y_targ) for func in self.metrics_funcs]
            _losses = np.array([loss, *metrics], dtype=np.object_)
        else:
            _losses = self.total_metrics.copy()
            _losses[0] = loss

        if self.comm is not None and comm:
            assert MPI

            _losses /= self.comm_size
            if blocking:
                _losses = self.comm.allreduce(_losses, op=MPI.SUM)
            else:
                loss_req = self.comm.iallreduce(_losses, op=MPI.SUM)
        else:
            if blocking:
                pass
            else:
                raise NotImplementedError("can not compute metrics non-blocking locally")

        return _losses, loss_req

    def _update_running_average(self, curr: np.ndarray, total: np.ndarray, count: int,
                                batch_size: int, prefix="") -> tuple[np.ndarray, int, str]:
        string = ""
        total = ((curr * batch_size) + (total * count)) / (count + batch_size)
        for c in range(len(self.loss_and_metrics)):
            loss_str = self.loss_and_metrics_format[c]
            if loss_str:
                string += ("%s, " % (prefix + loss_str)) % total[c]
        string = string[:-2]
        return total, count + batch_size, string

    def _model_reduce_sync(self, gradient=True):
        for layer in self.layers:
            self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW)
            layer.reduce_weights_sync(gradient=gradient)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_reduce_async(self, gradient=True):
        for layer in self.layers:
            self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.ALLREDUCE_DW)
            layer.reduce_weights_async(gradient=gradient)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    def _model_reduce_wait(self, gradient=True):
        for layer in self.layers:
            self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.WAIT_DW)
            layer.wait_allreduce_async(gradient=gradient)
            self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

    # TODO: Modify the method's name.
    def _weight_update(self, gradient=True, blocking=True, pipeline=False):
        if blocking:
            self._model_reduce_sync(gradient)
        elif pipeline:
            self._model_reduce_async(gradient)
            self._model_reduce_wait(gradient)
        else:
            self._model_reduce_wait(gradient)
            self._model_reduce_async(gradient)

    def train(self, bar_width=BAR_WIDTH) -> dict[str, list[np.ndarray]]:
        self._ensure_model_runable()

        # If working with CUDA, self.y_batch must be in a GPU's data structure.
        if self.enable_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)
            self.y_batch = tensor_ary  # type: ignore

        self.history = {lm: [] for lm in (self.loss_and_metrics + [f"val_{m}" for m in self.loss_and_metrics])}

        self.comm_nsamples = list(zip(*self.comm.allgather(self.dataset._nsamples) if self.comm else [self.dataset._nsamples]))

        terminate = False  # True: ends the following loop.
        global_terminate = False

        model_sync_count = 0
        train_batches_min = min(self.comm_nsamples[Dataset.Part.TRAIN]) / (self.batch_size * self.nprocs)
        val_batches_min = min(self.comm_nsamples[Dataset.Part.VAL]) / (self.batch_size * self.nprocs)

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

            for sched in self.schedulers:
                sched.on_epoch_begin(self, self.rank)

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

                self.rank_weight = self._compute_rank_weight(rank_mask, Dataset.Part.TRAIN)

                tic = timer()
                train_batch_loss = self._train_batch(x_batch, y_batch, sync_model=sync_model)
                toc = timer()

                if local_batch_size <= 0:
                    if self.comm_rank == 0:
                        pbar.set_postfix_str(s=f"{string}, waiting…", refresh=True)
                    continue

                train_total_loss, train_batch_count, string = \
                    self._update_running_average(train_batch_loss, train_total_loss, train_batch_count, batch_size)
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

                val_batch_loss = self._evaluate_batch(x_batch, y_batch, sync_model=sync_model)

                if batch_size <= 0:
                    continue

                val_total_loss, val_batch_count, string = \
                    self._update_running_average(val_batch_loss, val_total_loss, val_batch_count, batch_size, prefix="val_")
                if self.comm_rank == 0:
                    pbar.set_postfix_str(s=f"{train_string}, {string}", refresh=True)

            if self.comm_rank == 0:
                for c in range(len(self.loss_and_metrics)):
                    self.history["val_" + self.loss_and_metrics[c]].append(val_total_loss[c])

            for sched in self.schedulers:
                sched.on_epoch_end(train_total_loss, val_total_loss)
                if sched.stop_training:
                    terminate = True

            if self.comm_rank == 0:
                pbar.close()
                # Sleep for half a second to allow pbar to write its output before returning
                time.sleep(.5)

            for c in range(len(self.loss_and_metrics)):
                if not self.loss_and_metrics_format[c]:
                    logger.info(f"{self.loss_and_metrics[c]}: {train_total_loss[c]}")
            for c in range(len(self.loss_and_metrics)):
                if not self.loss_and_metrics_format[c]:
                    logger.info(f"val_{self.loss_and_metrics[c]}: {val_total_loss[c]}")

            if sync_epoch:
                if self.comm is not None:
                    op = MPI.LAND  # type: ignore
                    global_terminate = self.comm.allreduce(terminate, op=op)
                else:
                    global_terminate = terminate

            if global_terminate:
                break

        # Synchronize model
        if self.final_model_sync:
            self._weight_update(gradient=False, blocking=self.blocking_mpi)

        self.tracer.define_event_types(self)
        return self.history

    def _train_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model=True) -> np.ndarray:
        self.mode = Model.Mode.TRAIN

        # Schedulers begin
        for sched in self.schedulers:
            sched.on_batch_begin()

        self.real_batch_size = x_batch.shape[0]
        x, y_targ = self.layers[0]._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        if has_batch:
            # Forward pass (FP)
            for layer in self.layers:
                self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x = layer.forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            loss, dx = self.loss_func.compute(x, y_targ, self.real_batch_size)
        else:
            if y_targ.shape[0] != x_batch.shape[0]:
                raise ValueError(f"y_targ.shape[0] ({y_targ.shape[0]}) and x_batch.shape[0] ({x_batch.shape[0]}) must have the same value.")
            loss, dx = 0.0, y_targ

        total_metrics, _ = self._compute_metrics_funcs(x, y_targ, loss, comm=sync_model)
        assert total_metrics is not None
        self.total_metrics = total_metrics

        if has_batch:
            # Backward pass (BP)
            for layer in reversed(self.layers):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx = layer.backward(dx)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        if self.enable_cudnn:
            assert self.stream
            self.stream.synchronize()  # type: ignore

        # Gradient update (GU)
        if self.model_sync_freq >= 0 and sync_model:
            self._weight_update(gradient=True, blocking=self.blocking_mpi, pipeline=self.parallel_pipeline)

        if has_batch or sync_model:

            # Optimizer
            for layer in self.layers:
                self.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.UPDATE_DW)
                layer.update_weights(self.optimizer)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

        # Weight update (WU)
        if self.model_sync_freq > 0 and sync_model:
            self._weight_update(gradient=False, blocking=self.blocking_mpi, pipeline=self.parallel_pipeline)

        if self.enable_cudnn:
            for layer in self.layers:
                if layer.grad_vars:
                    layer.stream_2.synchronize()  # type: ignore

        # Schedulers end
        for sched in self.schedulers:
            sched.on_batch_end(self)

        return self.total_metrics

    def _compute_rank_weight(self, mask: list[int], part: Dataset.Part) -> float:
        match self.model_sync_participation:
            case Model.SyncParticipation.ALL:
                comm_nsamples = self.comm_nsamples[part]
            case Model.SyncParticipation.AVAIL2ALL:
                if mask[self.comm_rank]:
                    comm_nsamples = [nsamples for nsamples, mask in zip(self.comm_nsamples[part], mask) if mask]
                else:
                    return 0.0
            case _:
                raise ValueError(f"Model synchronization participation option '{self.model_sync_participation}' not recognized. Only recognized: {list(Model.SyncParticipation)}")

        min_nsamples, max_nsamples, total_nsamples = min(comm_nsamples), max(comm_nsamples), sum(comm_nsamples)
        comm_size = len(comm_nsamples)

        match self.model_sync_algo:
            case Model.SyncAlgorithm.AVG:
                return 1.0 / comm_size
            case Model.SyncAlgorithm.WAVG:
                return self.dataset._nsamples[part] / total_nsamples
            case Model.SyncAlgorithm.INVAVG:
                inverse_nsamples = min_nsamples + (max_nsamples - self.dataset._nsamples[part])
                return inverse_nsamples / total_nsamples
            case _:
                raise ValueError(f"Model synchronization algorithm option '{self.model_sync_algo}' not recognized. Only recognized: {list(Model.SyncAlgorithm)}")

    def _evaluate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray, sync_model=True) -> np.ndarray:
        self.mode = Model.Mode.EVALUATE

        self.real_batch_size = x_batch.shape[0]
        x, y_targ = self.layers[0]._sync_x_y(x_batch, y_batch)

        has_batch = x_batch.shape[0] > 0

        # Forward pass (FP)
        if has_batch:
            for i in range(len(self.layers)):
                self.tracer.emit_event(PYDTNN_MDL_EVENT, self.layers[i].id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                x = self.layers[i].forward(x)
                self.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)

            y_pred = self.layers[-1].y
            loss, _ = self.loss_func.compute(y_pred, y_targ, self.real_batch_size)
        else:
            y_pred = self.layers[-1].y
            loss = 0.0
        assert y_pred is not None

        total_metrics, _ = self._compute_metrics_funcs(y_pred, y_targ, loss, comm=sync_model)
        assert total_metrics is not None
        self.total_metrics = total_metrics

        return self.total_metrics

    def evaluate(self, bar_width=BAR_WIDTH):
        self._ensure_model_runable()

        if self.enable_cudnn and self.y_batch is None:
            assert gpuarray and self.cudnn_dtype
            tensor_ary = TensorArray(
                gpuarray.empty((self.batch_size, *self.layers[-1].shape), self.dtype),
                self.tensor_format, self.cudnn_dtype)
            self.y_batch = tensor_ary  # type: ignore

        self.comm_nsamples = list(zip(*self.comm.allgather(self.dataset._nsamples) if self.comm else [self.dataset._nsamples]))

        test_batches_min = min(self.comm_nsamples[Dataset.Part.TEST]) / (self.batch_size * self.nprocs)

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
                    self._update_running_average(test_batch_loss, test_total_loss, test_batch_count, batch_size, prefix="test_")
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
