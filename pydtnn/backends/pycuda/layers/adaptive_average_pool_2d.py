import math
from pydtnn.utils.tensor import TensorFormat
import numpy as np
from pycuda.driver import Function   # type: ignore
from pycuda.compiler import SourceModule   # type: ignore
from pycuda import gpuarray   # type: ignore
from pydtnn.utils.performance_models import im2col_time, col2im_time
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
import logging
logger = logging.getLogger(__name__)


# Import from AbstractPool2DLayerPycuda


class AdaptiveAveragePool2DPycuda(AdaptiveAveragePool2D[TensorArray], LayerPycuda):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # NOTE: Will be initalized later.
        self.y = None  # type: ignore

    def _model_init(self, prev_shape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        self.cuda_fwd_func = self._fwd_kernel()
        self.cuda_bwd_func = self._bwd_kernel()

        self.initialize_pool_2d_gpu(prev_shape, x)

    def initialize_pool_2d_gpu(self, prev_shape, x):
        self.ci, self.hi, self.wi = self.model.decode_shape(prev_shape)
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))
        pooling_shape = self.model.encode_shape((self.co, self.ho, self.wo))
        y = gpuarray.zeros((self.model.batch_size, *pooling_shape), self.model.dtype)
        self.y: TensorArray = TensorArray(y, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(self.x.ary.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

        self.fwd_time = \
            im2col_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (it's fine)
        self.bwd_time = \
            col2im_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (it's fine)

    def forward(self, x: TensorArray) -> TensorArray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)

        if self.pooling_not_needed:
            self.y = x
        else:
            n, c, h, w = self.model.decode_shape(x.shape)

            # NOTE: "num_elements" (or simply "N") is the number of elements to process. Usually it would be math.prod(x.shape),
            #   but in this case we are putting elements in the output instead of processing the input's elements.
            num_elements = np.int32(math.prod((n, c, self.ho, self.wo)))

            total_num_threads = np.int32(math.prod(self.grid) * math.prod(self.block))

            # If num_elements < total_num_threads, only will work "num_elements" threads. In the other cases will work "total_num_threads" threads.
            num_active_workers = np.int32(min(total_num_threads, num_elements))
            num_ops_per_worker = np.int32((num_elements + num_active_workers - 1) / num_active_workers)
            num_ops_last_worker = np.int32(num_elements - (num_active_workers - 1) * num_ops_per_worker)

            # NOTE: Instead of a number, PyCuda's driver expects "numpy.number"
            self.cuda_fwd_func(x.ary, self.y.ary,
                               np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                               np.int32(self.ho), np.int32(self.wo), num_elements,
                               num_active_workers, num_ops_per_worker, num_ops_last_worker,
                               grid=self.grid, block=self.block,
                               stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        n, c, h, w = self.model.decode_shape(dy.shape)

        num_elements = np.int32(math.prod((n, c, self.ho, self.wo)))

        total_num_threads = np.int32(math.prod(self.grid) * math.prod(self.block))

        num_active_workers = np.int32(min(total_num_threads, num_elements))
        num_ops_per_worker = np.int32((num_elements + num_active_workers - 1) / num_active_workers)
        num_ops_last_worker = np.int32(num_elements - (num_active_workers - 1) * num_ops_per_worker)
        self.dx.fill(0)

        self.cuda_bwd_func(self.dx.ary, self.y.ary,
                           np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                           np.int32(self.ho), np.int32(self.wo), num_elements,
                           num_active_workers, num_ops_per_worker, num_ops_last_worker,
                           grid=self.grid, block=self.block,
                           stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
