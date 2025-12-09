from pydtnn.layers.abstract.pool_2d_layer import AbstractPool2DLayer
from pydtnn.libs import libcudnn as cudnn
import pycuda.gpuarray as gpuarray   # type: ignore

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.performance_models import im2col_time, col2im_time
from pydtnn.layers.layer import ParameterException
from pydtnn.utils.constants import ArrayShape


class AbstractPool2DLayerGPU(AbstractPool2DLayer[TensorGPU], LayerGPU):
    """
    Provides common methods to Pool2DGPU classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attributes will be initalized later.
        self.pool_desc = None # TODO: set CDNN descripor type
        self.ci: int = None  # type: ignore
        self.hi: int = None  # type: ignore
        self.wi: int = None  # type: ignore
        self.kh: int = None  # type: ignore
        self.kw: int = None  # type: ignore
        self.co: int = None  # type: ignore
        self.ci: int = None  # type: ignore
        self.ho: int = None  # type: ignore
        self.wo: int = None  # type: ignore

    def initialize_pool_2d_gpu(self, prev_shape: ArrayShape, x: TensorGPU, pool_mode: int) -> None:
        # pool_mode comes from cudnn.CudnnPoolingMode

        if not (self.vdilation == 1 and self.hdilation == 1):
            raise ParameterException(f"cuDNN does not support dilated pooling. vdilation: {self.vdilation}, hdilation: {self.hdilation}")

        nan_prop = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']

        self.pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(self.pool_desc, pool_mode, nan_prop,
                                          self.kh, self.kw, self.vpadding, self.hpadding,
                                          self.vstride, self.hstride)
        # Get output dimensions
        _, _, self.ho, self.wo = cudnn.cudnnGetPooling2dForwardOutputDim(self.pool_desc, x.desc)
        self.shape = self.model.encode_shape((self.co, self.ho, self.wo))

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(self.x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            im2col_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            col2im_time(m=(self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo * self.ci),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)

    def forward(self, x: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        cudnn.cudnnPoolingForward(self.model.cudnn_handle, self.pool_desc, alpha,
                                  x.desc, x.ptr, beta,
                                  self.y.desc, self.y.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx
        cudnn.cudnnPoolingBackward(self.model.cudnn_handle, self.pool_desc, alpha,
                                   self.y.desc, self.y.ptr,
                                   dy.desc, dy.ptr,
                                   self.x.desc, self.x.ptr,
                                   beta, self.dx.desc, self.dx.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
