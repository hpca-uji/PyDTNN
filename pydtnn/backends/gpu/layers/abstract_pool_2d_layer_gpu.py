from abc import ABC

from pydtnn.layers import AbstractPool2DLayer
from ..libs import libcudnn as cudnn
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from . import LayerGPU
from ..tensor_gpu import TensorGPU
from pydtnn.performance_models import im2col_time, col2im_time
from pydtnn.utils import decode_tensor, encode_tensor


class AbstractPool2DLayerGPU(LayerGPU, AbstractPool2DLayer, ABC):
    """
    Provides common methods to Pool2DGPU classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_desc = None
        self.ci = self.hi = self.wi = self.kh = self.kw = self.co = self.ci = None
        self.ho = self.wo = None

    def initialize_pool_2d_gpu(self, prev_shape: tuple[int, ...], x: TensorGPU, pool_mode: cudnn.CudnnPoolingMode) -> None:
        self.hi, self.wi, self.ci = decode_tensor(prev_shape, self.model.tensor_format)
        if self.pool_shape[0] == 0:
            self.pool_shape = (self.hi, self.pool_shape[1])
        if self.pool_shape[1] == 0:
            self.pool_shape = (self.pool_shape[0], self.wi)
        self.kh, self.kw = self.pool_shape
        self.co = self.ci
        assert self.vdilation == 1 and self.hdilation == 1, "cuDNN does not support dilated pooling"

        nan_prop = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']

        self.pool_desc = cudnn.cudnnCreatePoolingDescriptor()
        cudnn.cudnnSetPooling2dDescriptor(self.pool_desc, pool_mode, nan_prop,
                                          self.kh, self.kw, self.vpadding, self.hpadding,
                                          self.vstride, self.hstride)
        # Get output dimensions
        _, _, self.ho, self.wo = cudnn.cudnnGetPooling2dForwardOutputDim(self.pool_desc, x.desc)
        self.shape = encode_tensor((self.ho, self.wo, self.co), self.model.tensor_format)

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
