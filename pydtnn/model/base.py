"""
Base module for PyDTNN models.

This module defines the abstract Base class that serves as the foundation for all
neural network models within the PyDTNN framework, providing common attributes,
memory management, and infrastructure for distributed training.
"""

import enum
import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from pydtnn import MPI_MODULE, Cublas_Handle_Type, Cudnn_Handle_Type
from pydtnn.abstract.layerable import Layerable
from pydtnn.losses.abstract.loss import Loss
from pydtnn.metrics.abstract.metric import Metric
from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.utils.constants import Array, ArrayShape, NetworkAlgoEnum
from pydtnn.utils.memory_pool import PrivateMemory
from pydtnn.utils.tensor import TensorFormat
from pydtnn.tracers.tracer import Tracer

__all__ = ("Base",)

logger = logging.getLogger(__name__)


# NOTE: mpi4py has more functions, but no typing
if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM  # type: ignore
else:
    MPI_COMM = ModuleType


class Base[T: Array]:
    """
    Base class for all models in PyDTNN.

    Provides the structural interface and shared state management for neural
    network architectures, including distributed training support and memory
    handling.
    """

    def __init__(self, **kwargs):
        """Initialize the base model instance."""
        pass

    class Mode(enum.StrEnum):
        """Enumeration for model execution modes."""

        EVALUATE = enum.auto()
        TRAIN = enum.auto()

    class SyncParticipation(enum.StrEnum):
        """Defines strategies for node participation in model synchronization."""

        ALL = enum.auto()
        AVAIL2ALL = enum.auto()

    class SyncAlgorithm(enum.StrEnum):
        """Defines algorithms for weight aggregation during synchronization."""

        AVG = enum.auto()
        WAVG = enum.auto()
        INVAVG = enum.auto()

    # Explicit declaration of those model attributes that are referenced by other parts of PyDTNN
    #   NOTE: The following parameters come from "Parser"
    dtype: np.dtype
    quantize: bool
    quantize_dtype: np.dtype
    num_epochs: int
    steps_per_epoch: float
    evaluate: bool
    evaluate_only: bool
    model_state_filename: str
    history_file: str
    tensor_format: TensorFormat
    random: np.random.Generator
    random_seed: int
    shared_tmp_memory: bool
    shared_data: bool
    model_sync_freq: int
    model_sync_algo: str
    model_sync_participation: str
    model_sync_min_avail: int
    initial_model_sync: bool
    final_model_sync: bool
    model_sync_quantize: bool
    model_sync_dtype: np.dtype
    dataset_name: str
    dataset_percentage: float
    dataset_path: str
    dataset_lang: str
    dataset_lang2: str
    synthetic_train_samples: int
    synthetic_test_samples: int
    synthetic_input_shape: str
    synthetic_output_shape: str
    test_as_validation: bool
    validation_split: float
    augment_shuffle: bool
    augment_horizontal_flip: float
    augment_vertical_flip: float
    augment_rotate: float
    augment_rotate_degree: float
    augment_brightness: float
    augment_brightness_factor: float
    augment_contrast: float
    augment_contrast_factor: float
    augment_saturation: float
    augment_saturation_factor: float
    augment_mask: float
    augment_mask_size: int
    augment_blur: float
    augment_blur_size: int
    augment_crop: bool
    augment_crop_perc: float
    augment_scale: bool
    augment_scale_size: int
    augment_perspective: float
    augment_perspective_factor: float
    augment_normalize: bool
    augment_normalize_offset: float
    augment_normalize_scale: float
    enable_fused_bn_relu: bool
    enable_fused_conv_relu: bool
    enable_fused_conv_bn: bool
    enable_fused_conv_bn_relu: bool
    conv_direct_method: str
    optimizer_name: str
    learning_rate: float
    learning_rate_scaling: bool
    optimizer_momentum: float
    optimizer_decay: float
    optimizer_nesterov: bool
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_epsilon: float
    optimizer_rho: float
    optimizer_tau: int
    optimizer_tau_prime: int
    optimizer_density: float
    loss_func_name: str
    loss_eps: float
    loss_weights: list[float] | None
    use_loss_weights: bool
    metrics: str
    schedulers_names: str
    early_stopping_metric: str
    early_stopping_patience: int
    early_stopping_minimize: bool
    reduce_lr_on_plateau_metric: str
    reduce_lr_on_plateau_float: float
    reduce_lr_on_plateau_patience: int
    reduce_lr_on_plateau_min_lr: float
    reduce_lr_every_nepochs_float: float
    reduce_lr_every_nepochs_nepochs: int
    reduce_lr_every_nepochs_min_lr: float
    stop_at_loss_metric: str
    stop_at_loss_threshold: float
    model_checkpoint_metric: str
    model_checkpoint_save_freq: int
    parallel_data: bool
    parallel_pipeline: bool
    use_blocking_mpi: bool
    use_mpi_buffers: bool
    enable_cudnn: bool
    enable_gpudirect: bool
    enable_nccl: bool
    enable_cudnn_auto_conv_algo: bool
    encryption_name: str
    encryption_slots: int
    encryption_scale: int
    encryption_security: int
    tracing: bool
    tracer: Tracer
    tracer_output: str
    tracer_pmlib_server: str
    tracer_pmlib_port: int
    tracer_pmlib_device: str
    profile: bool
    cpu_speed: float
    memory_bw: float
    network_bw: float
    network_lat: float
    network_algo: NetworkAlgoEnum
    mpi_processes: int
    threads_per_process: int
    gpus_per_node: int
    mpi_protocol: str
    mpi_server: str
    mpi_port: int

    kwargs: dict[str, Any]
    backend: str
    model_name: str
    history: dict[str, list[np.ndarray]]
    nparams: int

    cudnn_dtype: int
    cuda_grid: tuple[int, int, int]
    cuda_block: tuple[int, int, int]
    cudnn_handle: Cudnn_Handle_Type | None
    cublas_handle: Cublas_Handle_Type | None
    gpudirect: bool
    nccl_comm: Any | None
    nccl_type: Any | None
    stream: Any  # drv.Stream

    memory_used: int
    use_memory_pool: bool
    memory_cls: type[PrivateMemory]
    memory: PrivateMemory
    tmp_memory_used: int

    total_metrics: np.ndarray
    metrics_funcs: list[Metric[T]]
    loss_and_metrics: list[str]
    layers: list[Layerable]

    nprocs: int
    blocking_mpi: bool
    MPI: MPI_MODULE | None
    comm: MPI_COMM | None
    comm_size: int
    comm_rank: int
    comm_nsamples: list[tuple[int]]
    rank: int
    rank_weight: float

    batch_size: int
    global_batch_size: int
    real_batch_size: int
    input_shape: ArrayShape
    output_shape: ArrayShape

    evaluate_on_train: bool
    dataset_train_path: str
    dataset_test_path: str
    use_synthetic_data: bool
    y_batch: T
    optimizer: Optimizer[T]
    loss_func: Loss[T]
    _is_model_init: bool
