# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.elementwise import ElementwiseKernel

from pydtnn.layers.concatenation_block import ConcatenationBlock, CONCAT_DIM_NCHW, CONCAT_DIM_NHWC
from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, \
    PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers import LayerGPU
from pydtnn.backends.gpu.libs import libcudnn as cudnn
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils import decode_tensor, PYDTNN_TENSOR_FORMAT
from pydtnn.layers.layer import LayerError


class ConcatenationBlockGPU(LayerGPU, ConcatenationBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat = None
        self.split = None
        self.dy = None
        self.out_co = None
        self.idx_co = None

    def initialize(self, prev_shape: tuple[int, ...], x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)
        # @warning: super().initialize() calls self.initialize_block_layer() (don't call it again)
        self.concat = ElementwiseKernel(
            "T *dst, T *src, int N, int H, int W, int C, int first_c, int last_c".replace("T",
                                                                                          {np.float32: "float",
                                                                                           np.float64: "double"}[
                                                                                              self.model.dtype]),
            {PYDTNN_TENSOR_FORMAT.NHWC:
                """int c_ = i % C;
                   if (first_c <= c_ && c_ < last_c) {
                       int w_ = i / C % W;
                       int h_ = i / (W*C) % H;
                       int n_ = i / (H*W*C) % N;
                       int i_ = n_ * H * W * (last_c-first_c) + h_ * W * (last_c-first_c) + w_ * (last_c-first_c) + (c_-first_c);
                       dst[i] = src[i_];
                   }
                """,
             PYDTNN_TENSOR_FORMAT.NCHW:
                """int c_ = i / (H*W) % C;
                   if (first_c <= c_ && c_ < last_c) {
                       int w_ = i % W;
                       int h_ = i / W % H;
                       int n_ = i / (C*H*W) % N;
                       int i_ = n_ * (last_c-first_c) * H * W + (c_-first_c) * H * W + h_ * W + w_;
                       dst[i] = src[i_];
                   }
                """}[self.model.tensor_format],
            "concat")

        self.split = ElementwiseKernel(
            "T *src, T *dst, int N, int H, int W, int C, int first_c, int last_c".replace("T",
                                                                                          {np.float32: "float",
                                                                                           np.float64: "double"}[
                                                                                              self.model.dtype]),
            {PYDTNN_TENSOR_FORMAT.NHWC:
                """int c_ = i % C;
                   if (first_c <= c_ && c_ < last_c) {
                       int w_ = i / C % W;
                       int h_ = i / (W*C) % H;
                       int n_ = i / (H*W*C) % N;
                       int i_ = n_ * H * W * (last_c-first_c) + h_ * W * (last_c-first_c) + w_ * (last_c-first_c) + (c_-first_c);
                       dst[i_] = src[i];
                   }
                """,
             PYDTNN_TENSOR_FORMAT.NCHW:
                """int c_ = i / (H*W) % C;
                   if (first_c <= c_ && c_ < last_c) {
                       int w_ = i % W;
                       int h_ = i / W % H;
                       int n_ = i / (C*H*W) % N;
                       int i_ = n_ * (last_c-first_c) * H * W + (c_-first_c) * H * W + h_ * W + w_;
                       dst[i_] = src[i];
                   }
                """}[self.model.tensor_format],
            "split")

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # Derivative dy
        self.dy = []
        for i, p in enumerate(self.paths):
            dy_gpu = gpuarray.empty((self.model.batch_size, *self.out_shapes[i]), self.model.dtype)
            self.dy.append(TensorGPU(dy_gpu, self.model.tensor_format, self.model.cudnn_dtype))

    def forward(self, x: TensorGPU) -> TensorGPU:
        for i, p in enumerate(self.paths):
            y_i = x
            for layer in p:
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.FORWARD)
                y_i = layer.forward(y_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONCAT)
            self.concat(self.y.ary, y_i.ary, self.model.batch_size, self.ho, self.wo, self.co,
                        0 if i == 0 else self.idx_co[i - 1], self.idx_co[i])
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        for i, p in enumerate(self.paths):
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SPLIT)
            self.split(dy.ary, self.dy[i].ary, self.model.batch_size, self.ho, self.wo, self.co,
                       0 if i == 0 else self.idx_co[i - 1], self.idx_co[i])
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
            dx_i = self.dy[i]
            for layer in reversed(p):
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + PYDTNN_MDL_EVENT_enum.BACKWARD)
                dx_i = layer.backward(dx_i)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            if i == 0:
                self.dx = dx_i
            else:
                alpha, beta = 1.0, 1.0
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                             self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_ELTW_SUM)
                cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, dx_i.desc,
                                     dx_i.ptr, beta, self.dx.desc, self.dx.ptr)
                self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
