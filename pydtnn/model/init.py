"""Initialization code for the PyDTNN model"""

import itertools
import logging
from collections import abc
from typing import TYPE_CHECKING, Any

import numpy as np

from pydtnn import (MPI, context, cublas, cublas_handle, cudnn, cudnn_handle, drv,
                    gpuarray, hostname, nccl, nccl_comm, num_gpus, ranks_per_node, stream)
from pydtnn.abstract.layerable import Layerable
from pydtnn.datasets import select as select_dataset
from pydtnn.datasets.abstract import Dataset
from pydtnn.libs.mpi.rc import proto as proto
from pydtnn.losses import select as select_loss
from pydtnn.metrics import select as select_metric
from pydtnn.model.base import Base
from pydtnn.model.layers import Layers
from pydtnn.model.utils import DEFAULT_BACH_SIZE, LIMIT_THREADS_AND_BLOCKS
from pydtnn.models import select as select_model
from pydtnn.optimizers import select as select_optimizer
from pydtnn.utils import rand
from pydtnn.utils.gpu import CudnnDataType
from pydtnn.utils.memory_pool import PreallocMemory, PrivateMemory
from pydtnn.utils.performance_counter import PerformanceCounter
from pydtnn.utils.tensor import SampleFormat, TensorFormat, format_reshape

if TYPE_CHECKING:
    import polyhe  # type: ignore (polyhe exist if it's installed)
else:
    try:
        import polyhe
    except Exception:
        polyhe = None

from pydtnn.utils.constants import Array, NetworkAlgoEnum

__all__ = ("Init",)

logger = logging.getLogger(__name__)


class Init[T: Array](Layers[T]):  # noqa: D101 (generics not detected)
    """
    Main initialization class for PyDTNN models.

    Handles configuration of hardware backends (CUDA/MPI), memory management,
    dataset loading, and model layer construction.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the model environment and configuration.

        Args:
            **kwargs: Configuration arguments to override default parser settings.
        """
        super().__init__(**kwargs)

        # Filter default values from base and update them from the received kwargs
        for key, value in vars(Base).items():
            if key.startswith("_"):
                continue
            self.__dict__[key] = kwargs.get(key, value)

        # Attributes related to the given arguments
        self.blocking_mpi: bool = self.use_blocking_mpi
        self.use_cudnn = gpuarray is not None and drv is not None and cublas is not None
        self.gpudirect: bool = self.use_gpudirect
        self.use_nccl: bool = self.use_nccl
        self.memory: PrivateMemory = None  # type: ignore (it will be intialized later if "self.use_memory_pool" is True)
        self.dtype: np.dtype = np.dtype(self.dtype)
        self.param_dtype: np.dtype = np.dtype(self.quantize_dtype) if self.quantize else self.dtype
        self.network_algo = NetworkAlgoEnum(self.network_algo.lower())
        # self.metrics_dtype: np.dtype = np.dtype(np.float32) if np.issubdtype(self.dtype, np.int32) else self.dtype

        self.nparams = 0
        self.memory_used = 0
        self.tmp_memory_used = 0

        # Set performance counter
        self.perf_counter = PerformanceCounter()

        # Layers' attributes
        self.layers: list[Layerable[T]] = []
        self.layer_id_generator: abc.Iterator[int] = iter(itertools.count())

        # Set current mode to unspecified
        self.mode: Base.Mode = None  # type: ignore # Base.Mode.UNSPECIFIED
        self.random: np.random.Generator = rand.Generator(self.random_seed)  # type: ignore
        self.memory_cls = PreallocMemory if self.shared_tmp_memory else PrivateMemory

        # Set tracer
        self._tracer_init()

        # Data format
        self._tensor_init()
        self._batch_init()

        # Set MPI and comm
        self._mpi_init()

        # Encryption [NOTE: Always after initializing MPI (if you are going to use MPI)]
        if self.encryption_name:
            self.crypt = self._crypt_init(self.encryption_name)
        else:
            self.crypt = None

        # Cuda [NOTE: Always after initializing MPI (if you are going to use MPI)]
        if self.use_cudnn:
            self._cudnn_init()
        else:
            self.stream = None

        # Dataset [NOTE: Always after initializing MPI (if you are going to use MPI)]
        if self.dataset_name:
            self.dataset: Dataset = select_dataset(self.dataset_name)(self)

        # Private attributes
        self._is_model_init: bool = False

        # LR scaling
        if self.learning_rate_scaling or (
            self.learning_rate_scaling is None and self.global_batch_size is None
        ):
            self.learning_rate = self.learning_rate * self.nprocs

        # Optimizers
        self.optimizer = select_optimizer(self.optimizer_name).from_model(self)
        self.optimizer._init_backend_with_model(self)

        # Metrics list
        self.metrics_list: list[str] = [m for m in self.metrics]

        # Synchronization parameters
        # NOTE: This parameter come from Parser.
        self.model_sync_algo = self.SyncAlgorithm(self.model_sync_algo)

        # NOTE: This parameter come from Parser.
        self.model_sync_participation = self.SyncParticipation(self.model_sync_participation)

        # Read the model (NOTE: must be the last action, as it calls
        # self._model_init() if there is a model)
        if model_name := self.model_name:
            self._layers_init(model_name)

    def _tensor_init(self) -> None:
        """Configures the tensor format based on hardware capabilities."""
        if self.tensor_format:
            tensor_format = TensorFormat(self.tensor_format.lower())
        elif self.use_cudnn:
            tensor_format = TensorFormat.NCHW
        else:
            tensor_format = TensorFormat.NHWC

        self.tensor_format = tensor_format

    def _batch_init(self, default: int = DEFAULT_BACH_SIZE) -> None:
        """
        Configures the batch size for training.

        Args:
            default: Default batch size if none is provided.
        """
        if self.batch_size and self.global_batch_size:
            raise ValueError(
                "Can not define 'local_batch_size' and 'global_batch_size' simultaneously"
            )
        elif self.global_batch_size:
            # NOTE: Using comm_size instead of nprocs might not be appropriate,
            #       as it differs to how global_batch_size is defined elsewhere,
            #       but for now it just a parser option difference that helps testing
            batch_size = self.global_batch_size // self.comm_size
        elif self.batch_size:
            batch_size = self.batch_size
        else:
            batch_size = default

        if batch_size < 1:
            raise ValueError(
                f"'batch_size' ({batch_size}) "
                f"too small or too many processes (num processes: {self.comm_size})"
            )

        self.batch_size = batch_size

    def _tracer_init(self) -> None:
        """Initializes the performance tracing mechanism."""
        if self.tracer_output == "":
            from pydtnn.tracers.extrae_tracer import ExtraeTracer

            tracer = ExtraeTracer(self.tracing)
        elif self.use_cudnn:
            from pydtnn.tracers.simple_tracer_gpu import SimpleTracerPycuda

            tracer = SimpleTracerPycuda(self.tracing, self.tracer_output, self.comm)
        elif self.tracer_pmlib_device != "":
            from pydtnn.tracers.simple_tracer_pmlib import SimpleTracerPMLib

            tracer = SimpleTracerPMLib(
                self.tracing,
                self.tracer_output,
                self.comm,
                self.tracer_pmlib_server,
                self.tracer_pmlib_port,
                self.tracer_pmlib_device,
            )
        else:
            from pydtnn.tracers.simple_tracer import SimpleTracer

            tracer = SimpleTracer(self.tracing, self.tracer_output, self.comm)

        self.tracer = tracer

    def _crypt_init(self, encryption_name: str) -> "polyhe.Context":
        """
        Initializes the homomorphic encryption context.

        Args:
            encryption_name: Name of the encryption backend to use.
        """
        if polyhe is None:
            raise RuntimeError("uHE is not avaliable, but is requiested!")

        backend = polyhe.Backend(encryption_name)
        options = polyhe.Options(
            slots=self.encryption_slots,
            scale=self.encryption_scale,
            security=self.encryption_security,
        )

        if self.comm_rank == 0:
            crypt = polyhe.new(backend, options)

        if self.comm:
            crypt = self.comm.bcast(crypt if self.comm_rank == 0 else None)

        assert crypt is not None
        if self.use_nccl:
            logger.warning("If NCCL is active, encryption is disabled")

        return crypt

    def _mpi_init(self) -> None:
        """Initializes MPI communication settings and process ranks."""
        # Communication type
        if self.parallel_data or self.parallel_pipeline:
            if not MPI:
                raise ValueError("Please, install mpi4py to allow parallel MPI execution!")
            self.MPI, self.comm = (MPI, MPI.COMM_WORLD)
        else:
            self.MPI, self.comm = (None, None)  # type: ignore (Excepcionally, they can be "None")

        # Communication weight
        self.rank_weight = 1.0

        # Communication method
        match self.use_mpi_buffers:
            case None:
                self.use_mpi_buffers = proto is None
            case bool():
                pass
            case _:
                raise ValueError(f"MPI buffers option '{self.use_mpi_buffers}' not recognized.")

    def _cudnn_init(self) -> None:
        """Initializes CUDA, cuDNN, and NCCL backend handles."""

        if not self.gpudirect and self.use_nccl:
            raise RuntimeError("It is necessary to have gpudirect active to work with NCCL.")

        assert cudnn is not None
        assert drv is not None
        assert context is not None
        assert cudnn_handle is not None
        assert cublas_handle is not None
        assert stream is not None

        self.y_batch = None  # type: ignore (it will be initalized later)

        self.cuda_threads = min(self.batch_size, LIMIT_THREADS_AND_BLOCKS)
        self.cuda_blocks = (max(self.batch_size, LIMIT_THREADS_AND_BLOCKS) // self.cuda_threads) + 1
        # NOTE: Seems that in PyDTNN, usually the ".x" (blockIdx.x, threadIdx.x,
        # ...) is the only dimension used.
        self.cuda_grid = (self.cuda_blocks, 1, 1)
        self.cuda_block = (self.cuda_threads, 1, 1)

        if self.comm and self.use_nccl:
            assert nccl is not None
            assert nccl_comm is not None

            nccl_types = {
                np.float64: nccl.DataType.Float64,
                np.float32: nccl.DataType.Float32,
                np.int8: nccl.DataType.Int8,
                np.int32: nccl.DataType.Int32,
            }

            nccl_type = nccl_types.get(self.dtype, nccl.DataType.Float32)

            if ranks_per_node[hostname] > num_gpus:
                raise ValueError("Not able to run more processes than GPUs per node!")
        else:
            nccl_type = None

        self.tracer.set_stream(stream)

        cudnn_types = {
            np.float64: CudnnDataType.FLOAT64,
            np.float32: CudnnDataType.FLOAT32,
            np.int8: CudnnDataType.INT8,
            np.int32: CudnnDataType.INT32,
        }

        cudnn_type: str = cudnn_types.get(self.dtype, CudnnDataType.FLOAT32)
        cudnn_dtype: int = cudnn.cudnnDataType[cudnn_type]

        self.nccl_type = nccl_type
        self.nccl_comm = nccl_comm
        self.cudnn_handle = cudnn_handle
        self.cublas_handle = cublas_handle
        self.stream = stream
        self.cudnn_dtype = cudnn_dtype

    def _layers_init(self, model_name: str) -> None:
        """
        Initializes model layers from a predefined model name.

        Args:
            model_name: Name of the model architecture to instantiate.
        """
        create_model = select_model(model_name)
        input_shape = self.dataset.input_shape
        output_shape = self.dataset.output_shape

        # NOTE: Dataset is always in NCHW
        # Change input_shape to model.tensor_format
        if len(input_shape) != 3:
            logger.warning(
                f"Input layer does not have 3 dimensions ({input_shape}), it may cause issues!"
            )
        else:
            input_shape = format_reshape(
                input_shape, SampleFormat.CHW, self.tensor_format.as_sample()
            )

        self.add_layers(create_model(input_shape, output_shape))

    def _model_init(self) -> None:
        """Finalizes model initialization, including memory allocation and component setup."""
        if self._is_model_init:
            return
        self._is_model_init = True

        self.random.seed(self.random_seed)  # type: ignore (seed exists)

        self.dataset._model_init()

        self._apply_layer_fusion()

        temp_memory_size = []

        self.loss_func = select_loss(self.loss_func_name).from_model(self)
        self.loss_func._init_backend_with_model(self)
        self.loss_func._model_init()
        self.memory_used += self.loss_func.memory_used
        temp_memory_size.append(self.loss_func.tmp_memory_used)

        metrics = [(m, select_metric(m).from_model(self)) for m in self.metrics_list]
        metrics.sort(key=lambda metric: metric[1].order)
        self.metrics_list, self.metrics_funcs = map(list, zip(*metrics))

        for metric in self.metrics_funcs:
            metric._init_backend_with_model(self)
            metric._model_init()
            self.memory_used += metric.memory_used
            temp_memory_size.append(metric.tmp_memory_used)

        self.loss_and_metrics = [self.loss_func_name] + self.metrics_list
        self.loss_and_metrics_format = [self.loss_func.format] + [
            metric.format for metric in self.metrics_funcs
        ]
        self.total_metrics = np.array([0] + [0] * len(self.metrics_funcs), dtype=self.dtype)
        self.tracer.define_event_types(self)

        self.optimizer._model_init(self.get_all_layers(self.layers))
        self.memory_used += self.optimizer.memory_used
        temp_memory_size.append(self.optimizer.tmp_memory_used)

        for layer in self.layers:
            self.memory_used += layer.memory_used
            temp_memory_size.append(layer.tmp_memory_used)

        self.tmp_memory_used += self.memory_cls._total(*temp_memory_size)
        self.memory_used += self.tmp_memory_used
        self.memory = self.memory_cls(size=self.tmp_memory_used)

        for layer in self.get_all_layers():
            layer._post_init()

        for metric in self.metrics_funcs:
            metric._post_init()

        self.loss_func._post_init()
        self.optimizer._post_init()

    def _ensure_model_runnable(self) -> None:
        """Validates that the model is ready for execution."""
        if not self.layers:
            raise ValueError("The model has no layers in it.")
        elif not self.dataset:
            raise ValueError("There is no dataset and the model has layers.")
        self._model_init()
