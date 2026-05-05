import enum
import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from pydtnn import MPI_MODULE, Cublas_Handle_Type, Cudnn_Handle_Type
from pydtnn.abstract.layerable import Layerable
from pydtnn.losses.loss import Loss
from pydtnn.metrics.metric import Metric
from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array, ArrayShape, NetworkAlgEnum
from pydtnn.utils.memory_pool import PrivateMemory
from pydtnn.utils.tensor import TensorFormat

__all__ = (
    "Base",
)

logger = logging.getLogger(__name__)


# NOTE: mpi4py has more functions, but no typing
if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM  # type: ignore
else:
    MPI_COMM = ModuleType


class Base[T: Array]:
    def __init__(self, **kwargs):
        pass

    class Mode(enum.StrEnum):
        EVALUATE = enum.auto()
        TRAIN = enum.auto()

# Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
#   NOTE: The following parameters come from "Parser"
    backend: str
    tensor_format: TensorFormat
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
    blocking_mpi: bool
    # enable_memory_cache: bool
    enable_nccl: bool
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
    model_name: str
    global_batch_size: int
    dataset_path: str
    quantize: bool
    quantize_dtype: np.dtype
    enable_cudnn: bool
    batch_size: int
    layers: list[Layerable]
    kwargs: dict[str, Any]

    memory_cls: type[PrivateMemory]
    memory: PrivateMemory
    memory_used: int
    tmp_memory_used: int

    rank_weight: float
    comm_rank: int
    comm_size: int
    rank: int
    nprocs: int
    learning_rate: float
    MPI: MPI_MODULE | None
    comm: MPI_COMM | None

    gpudirect: bool
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
