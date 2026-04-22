from typing import Any
from warnings import warn

import numpy as np

from pydtnn import MPI, drv, nccl, cudnn
from pydtnn import hostname, ranks_per_node, num_gpus, nccl_comm, cudnn_handle, cublas_handle, context, stream

from pydtnn._model.model_layer import Model_Layer as Model
from pydtnn.libs.mpi.rc import proto as PROTOCOL
from pydtnn import MPI, utils
from pydtnn.losses.loss import select as select_loss
from pydtnn.metrics.metric import select as select_metric
from pydtnn.optimizers.optimizer import select as select_optimizer
from pydtnn.utils.gpu import CudnnDataType

import logging

from pydtnn.utils.constants import Array
logger = logging.getLogger(__name__)

class Model_Init[T: Array](Model[T]):
    
    LIMIT_THREADS_AND_BLOCKS = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set MPI and comm
        self._mpi_init()

        # Cuda
        if self.enable_cudnn:
            self._cudnn_init()

        # Read the model (must be the last action, as it calls self._model_init() if there is a model)
        self.model_name: str | None = self.kwargs.get("model_name")
        if self.model_name:
            self._read_model(self.model_name)
        
        # Private attributes
        self._is_model_init: bool = False

        # Optimizers and LRSchedulers
        if self.learning_rate_scaling:
            # using comm_size instead of nprocs might not be appropriate,
            # as it differs to how learning_rate is defined elsewhere,
            # but for now it just a parser option difference that helps testing
            self.learning_rate = self.learning_rate / self.comm_size

        self.optimizer = select_optimizer(self.optimizer_name).from_model(self)
        self.optimizer._init_backend_with_model(self)

        # Metrics list
        self.metrics_list: list[str] = [m for m in self.metrics.replace(" ", "").split(",")]
    # ----- __init__ ----- #

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
    # ----- _mpi_init ----- #

    def _cudnn_init(self) -> None:
        self.cuda_threads = min(self.batch_size, self.LIMIT_THREADS_AND_BLOCKS)
        self.cuda_blocks = (max(self.batch_size, self.LIMIT_THREADS_AND_BLOCKS) // self.cuda_threads) + 1
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

        cudnn_types = {np.float64: CudnnDataType.FLOAT64,
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
    # ----- _cudnn_init ----- #

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