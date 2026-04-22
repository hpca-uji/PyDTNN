import enum
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
import logging

from pydtnn.datasets.dataset import Dataset
from pydtnn.optimizers.optimizer import Optimizer
logger = logging.getLogger(__name__)

import itertools

from pydtnn.abstract.layerable import Layerable
from pydtnn.metrics.metric import Metric

from pydtnn.losses.loss import Loss
from pydtnn.parser import PydtnnArgumentParser
from pydtnn.utils.constants import Array, ArrayShape, NetworkAlgEnum, Parameters
from pydtnn.utils.tensor import TensorFormat, encode_shape, encode_tensor, decode_shape, decode_tensor
from pydtnn import MPI_MODULE, Cudnn_Handle_Type, Cublas_Handle_Type, drv, gpuarray, cublas
from collections import abc
from pydtnn.utils.performance_counter import PerformanceCounter
from pydtnn.utils.memory_pool import PrivateMemory, PreallocMemory

from pydtnn.context.utils import get_batch_size, get_tensor_format, get_tracer
from pydtnn.datasets.dataset import select as select_dataset

# NOTE: mpi4py has more functions, but no typing
if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM
else:
    MPI_COMM = ModuleType

class Context_Base[T: Array]:

    BAR_WIDTH = 140
    DEFAULT_BACH_SIZE = 64

    class Mode(enum.StrEnum):
        EVALUATE = enum.auto()
        TRAIN = enum.auto()
        UNSPECIFIED = enum.auto()
    # ---

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
    enable_fused_conv_bn: bool
    enable_fused_conv_relu: bool
    enable_fused_bn_relu: bool
    enable_fused_conv_bn_relu: bool
    conv_direct_method: str
    parallel_data: bool
    parallel_pipeline: bool
    use_blocking_mpi: bool
    enable_gpudirect: bool
    shared_tmp_memory: bool
    tracing: bool
    tracer_output: str
    tracer_pmlib_server: str
    tracer_pmlib_port: int
    tracer_pmlib_device: str
    model_name:str
    global_batch_size:int
    dataset_path: str
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
    optimizer: Optimizer
    
##########################################
    ## INIT ##
    ##########
    def __init__(self, **kwargs):

        # Get default values from parser and update them from the received kwargs
        self.kwargs: dict[str, Any] = PydtnnArgumentParser().get_default_values()
        self.kwargs.update(kwargs)

        # Attributes related to the given arguments
        self.blocking_mpi: bool = self.use_blocking_mpi # TODO: MIRAR de dónde sale esto.
        self.enable_cudnn = gpuarray is not None and drv is not None and cublas is not None
        self.gpudirect: bool = self.enable_gpudirect
        self.enable_nccl: bool = self.enable_nccl
        self.dtype: np.dtype = np.dtype(self.dtype)
        self.memory: PrivateMemory = None  # type: ignore (it will be intialized later if "self.use_memory_pool" is True)

        self.nparams = 0
        self.memory_used = 0
        self.tmp_memory_used = 0

        # Set performance counter
        self.perf_counter = PerformanceCounter()

        # Layers' attributes
        self.layers: list[Layerable] = []
        self.layer_id_generator: abc.Iterator[int] = iter(itertools.count())

        # Set current mode to unspecified
        self.mode: Context_Base.Mode = Context_Base.Mode.UNSPECIFIED


        self.memory_cls = PreallocMemory if self.shared_tmp_memory else PrivateMemory

        # Set tracer
        self.tracer = get_tracer(tracer_output=self.tracer_output, tracing=self.tracing, comm=self.comm, enable_cudnn=self.enable_cudnn,
                                 tracer_pmlib_server=self.tracer_pmlib_server, tracer_pmlib_port=self.tracer_pmlib_port,
                                 tracer_pmlib_device=self.tracer_pmlib_device)

        # Data format
        self.tensor_format: TensorFormat = get_tensor_format(tensor_format=self.tensor_format, gpu=self.enable_cudnn)

        self.batch_size = get_batch_size(local_size=self.batch_size, global_size=self.global_batch_size, comm_size=self.comm_size)

        # Load weights and bias
        if self.weights_and_bias_filename:
            self.load_weights_and_bias(self.weights_and_bias_filename)
        # Dataset
        if self.dataset_name:
            self.dataset: Dataset = select_dataset(self.dataset_name)(self)
    # ---- #


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
    
    def export(self) -> dict[str, Any]:
        data = {}

        if self.model_name is not None:
            data[Parameters.MODEL_NAME] = self.model_name

        data[Parameters.LAYERS] = [
            layer.export()
            for layer in self.layers
        ]

        return data

    def import_(self, data: "dict[str, Any] | Context_Base") -> None:
        if isinstance(data, Context_Base):
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
