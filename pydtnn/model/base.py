"""
Base module for PyDTNN models.

This module defines the abstract Base class that serves as the foundation for all
neural network models within the PyDTNN framework, providing common attributes,
memory management, and infrastructure for distributed training.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from pydtnn import MPI_MODULE, Cublas_Handle_Type, Cudnn_Handle_Type
from pydtnn.abstract.layerable import Layerable
from pydtnn.datasets.abstract import Dataset
from pydtnn.losses.abstract.loss import Loss
from pydtnn.metrics.abstract.metric import Metric
from pydtnn.optimizers.abstract.optimizer import Optimizer
from pydtnn.schedulers.abstract.scheduler import Scheduler
from pydtnn.tracers.tracer import Tracer
from pydtnn.utils.constants import Array, ArrayShape, NetworkAlgoEnum
from pydtnn.utils.memory_pool import PrivateMemory
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Base",)

logger = logging.getLogger(__name__)


# NOTE: mpi4py has more functions, but no typing
if TYPE_CHECKING:
    from pycuda.driver import Stream
    from pympi.MPI import Comm as MPI_COMM  # noqa: N814


class ModelMode(enum.StrEnum):
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


class Base[T: Array]:  # noqa: D101 (generics not detected)
    """
    Base class for all models in PyDTNN.

    Provides the structural interface and shared state management for neural
    network architectures, including distributed training support and memory
    handling.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the base model instance."""
        self.nparams = 0
        self.memory_used = 0
        self.tmp_memory_used = 0
        self.mode: ModelMode = None  # pyright: ignore[reportAttributeAccessIssue] # Mode.UNSPECIFIED

    # NOTE: typing only (NO DEFAULTS)
    config: dict
    _model_inited: bool
    comm_nsamples: tuple[tuple[int], ...]
    comm_rank: int
    comm_size: int
    comm: MPI_COMM
    cublas_handle: Cublas_Handle_Type
    cuda_block: tuple[int, int, int]
    cuda_grid: tuple[int, int, int]
    cudnn_dtype: int
    cudnn_handle: Cudnn_Handle_Type
    history: list[dict[str, Any]]
    input_shape: ArrayShape
    kwargs: dict[str, Any]
    layers: list[Layerable[T]]
    loss_and_metric_names: tuple[str, ...]
    loss_func: Loss[T]
    schedulers: list[Scheduler]
    memory_cls: type[PrivateMemory]
    memory_used: int
    memory: PrivateMemory
    metrics_funcs: tuple[Metric[T], ...]
    MPI: MPI_MODULE
    nccl_comm: Any
    nccl_type: Any
    nparams: int
    nprocs: int
    optimizer: Optimizer[T]
    output_shape: ArrayShape
    random: np.random.Generator
    rank_weight: float
    rank: int
    real_batch_size: int
    stream: Stream
    tmp_memory_used: int
    tracer: Tracer
    use_cuda: bool
    use_gpudirect: bool
    use_memory_pool: bool
    y_batch: T
    dataset: Dataset

    # NOTE: Kwargs defaults (DEFAULT REQUIRED)
    model_name: str = ""
    backend: str = "cpu"
    batch_size: int = 0
    global_batch_size: int = 0
    dtype: np.dtype = np.dtype(np.float32)
    quantize: bool = False
    quantize_dtype: np.dtype = np.dtype(np.float16)
    num_epochs: int = 1
    steps_per_epoch: int = 0
    evaluate_on_train: bool = False
    evaluate_only: bool = False
    model_state_file: str = ""
    logger: bool = True
    use_history: bool = False
    tensor_format: TensorFormat = None  # pyright: ignore[reportAssignmentType]
    random_seed: int = 57005
    shared_tmp_memory: bool = False
    shared_data: bool = True
    model_sync_freq: int = 0
    model_sync_algo: SyncAlgorithm = SyncAlgorithm.AVG
    model_sync_participation: SyncParticipation = SyncParticipation.ALL
    model_sync_min_avail: int = 0
    initial_model_sync: bool = True
    final_model_sync: bool = True
    model_sync_quantize: bool = False
    model_sync_dtype: np.dtype = np.dtype(np.float16)
    dataset_name: str = ""
    dataset_percentage: float = 0.0
    dataset_path: str = "datasets/mnist"
    dataset_lang: str = "en"
    dataset_lang2: str = "de"
    synthetic_train_samples: int = 1000
    synthetic_test_samples: int = 100
    synthetic_input_shape: ArrayShape = (3, 32, 32)
    synthetic_output_shape: ArrayShape = (10,)
    test_as_validation: bool = False
    validation_split: float = 0.2
    augment_shuffle: bool = True
    augment_horizontal_flip: float = 0.0
    augment_vertical_flip: float = 0.0
    augment_rotate: float = 0.0
    augment_rotate_degree: float = 90.0
    augment_brightness: float = 0.0
    augment_brightness_factor: float = 1.0
    augment_contrast: float = 0.0
    augment_contrast_factor: float = 1.0
    augment_saturation: float = 0.0
    augment_saturation_factor: float = 1.0
    augment_mask: float = 0.0
    augment_mask_size: int = 16
    augment_blur: float = 0.0
    augment_blur_size: int = 16
    input_crop: bool = False
    input_crop_perc: float = 0.875
    input_scale: bool = False
    input_scale_size: int = 300
    augment_perspective: float = 0.0
    augment_perspective_factor: float = 0.25
    input_normalize: bool = False
    input_normalize_offset: float = 0.0
    input_normalize_scale: float = 0.0
    fused_bn_relu: bool = False
    fused_conv_relu: bool = False
    fused_conv_bn: bool = False
    fused_conv_bn_relu: bool = False
    conv_direct_method: str = ""
    optimizer_name: str = "sgd"
    learning_rate: float = 1e-2
    learning_rate_scaling: bool = None  # pyright: ignore[reportAssignmentType]
    optimizer_momentum: float = 0.9
    optimizer_decay: float = 0.0
    optimizer_nesterov: bool = False
    optimizer_beta1: float = 0.99
    optimizer_beta2: float = 0.999
    optimizer_epsilon: float = 1e-7
    optimizer_rho: float = 0.9
    optimizer_tau: int = 64
    optimizer_tau_prime: int = 32
    optimizer_density: float = 0.01
    oktopk_min_k: int = 10
    oktopk_partition_method: str = "sparse"
    oktopk_reduce_method: str = "p2p_region_wise_reduce_destination_rotation_and_bucketing"
    loss_name: str = "negative_likelihood"
    loss_eps: float = 1e-8
    class_weights: tuple[float, ...] = ()
    use_class_weights: bool = False
    metric_names: tuple[str, ...] = ("categorical_accuracy",)
    schedulers_names: tuple[str, ...] = (
        "early_stopping",
        "reduce_lr_on_plateau",
        "model_checkpoint",
    )
    warm_up_epochs: int = 5
    early_stopping_metric: str = "val_negative_likelihood"
    early_stopping_patience: int = 10
    early_stopping_minimize: bool = True
    reduce_lr_on_plateau_metric: str = "val_negative_likelihood"
    reduce_lr_on_plateau_factor: float = 0.1
    reduce_lr_on_plateau_patience: int = 5
    reduce_lr_on_plateau_min_lr: float = 0.0
    reduce_lr_every_nepochs_factor: float = 0.1
    reduce_lr_every_nepochs_nepochs: int = 5
    reduce_lr_every_nepochs_min_lr: float = 0.0
    stop_at_loss_metric: str = "val_accuracy"
    stop_at_loss_threshold: float = 0
    model_checkpoint_metric: str = "val_negative_likelihood"
    model_checkpoint_save_freq: int = 2
    parallel_data: bool = False
    parallel_pipeline: bool = False
    use_blocking_mpi: bool = True
    use_mpi_buffers: bool = None  # pyright: ignore[reportAssignmentType]
    use_gpudirect: bool = False
    use_nccl: bool = False
    use_cudnn_auto_conv_algo: bool = True
    encryption_name: str = ""
    encryption_slots: int = 13
    encryption_scale: int = 40
    encryption_security: int = 128
    tracing: bool = False
    tracer_output: str = ""
    tracer_pmlib_server: str = "127.0.0.1"
    tracer_pmlib_port: int = 6526
    tracer_pmlib_device: str = ""
    profile: bool = False
    cpu_speed: float = 4e12
    memory_bw: float = 50e9
    network_bw: float = 1e9
    network_lat: float = 0.5e-6
    network_algo: NetworkAlgoEnum = NetworkAlgoEnum.VDG
