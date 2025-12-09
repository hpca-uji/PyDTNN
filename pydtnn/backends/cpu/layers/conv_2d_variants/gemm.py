import numpy as np

from pydtnn.backends.cpu.layers.abstract.conv_2d_standard import Conv2DStandardCPU
from pydtnn.libs.libconvgemm import ConvGemm
from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat


class Conv2DGemmCPU(Conv2DStandardCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convGemm related attributes (will be initialized in initialize())
        self.cg = None # type: ignore

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        # ConvGemm parameters
        self.cg: ConvGemm = ConvGemm(dtype=self.model.dtype, debug=self.debug, parent_layer=self)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_cg_nchw
                self.backward = self._backward_cg_nchw
            case TensorFormat.NHWC:
                self.forward = self._forward_cg_nhwc
                self.backward = self._backward_cg_nhwc
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        # ---
    # ----

    def _forward_cg_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm library"""

        self.cg_x = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        y: np.ndarray = self.cg.conv_gemm_nhwc(self.weights, x,
                                               vpadding=self.vpadding, hpadding=self.hpadding,
                                               vstride=self.vstride, hstride=self.hstride,
                                               vdilation=self.vdilation, hdilation=self.hdilation,
                                               biases=self.biases)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return y

    def _forward_cg_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses the convGemm library"""

        self.cg_x = x

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CONVGEMM)
        res = self.cg.conv_gemm_nchw(self.weights, x,
                                     vpadding=self.vpadding, hpadding=self.hpadding,
                                     vstride=self.vstride, hstride=self.hstride,
                                     vdilation=self.vdilation, hdilation=self.hdilation,
                                     biases=self.biases)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return res

    def _backward_cg_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses the convGemm library"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CONVGEMM)
        res: np.ndarray = np.zeros(self.weights.shape, dtype=dy.dtype)
        self.cg.conv_gemm_nhwc(dy, self.cg_x, out=res,
                               vpadding=self.vpadding, hpadding=self.hpadding,
                               vstride=self.vstride, hstride=self.hstride,
                               vdilation=self.vdilation, hdilation=self.hdilation,
                               trans=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        self.dw = res

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db: np.ndarray = np.sum(dy, axis=(0, 1, 2))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_DECONV_GEMM)
        dx: np.ndarray = np.zeros((dy.shape[0], self.hi, self.wi, self.ci), dtype=dy.dtype)
        self.cg.deconv_gemm_nhwc(self.weights, dy, dx,
                                 vpadding=self.vpadding, hpadding=self.hpadding,
                                 vstride=self.vstride, hstride=self.hstride,
                                 vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx

    def _backward_cg_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses the convGemm library"""
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CONVGEMM)
        res = np.zeros(self.weights.shape, dtype=dy.dtype)
        # NOTE: conv_gemm_nchw, in this context seems that is being used as a matrix multiplication instead of a convolution.
        self.cg.conv_gemm_nchw(dy, self.cg_x, out=res,
                               vpadding=self.vpadding, hpadding=self.hpadding,
                               vstride=self.vstride, hstride=self.hstride,
                               vdilation=self.vdilation, hdilation=self.hdilation,
                               trans=True)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        self.dw = res

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
        if self.use_bias:
            self.db = np.sum(dy, axis=(0, 2, 3))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                     self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_DECONV_GEMM)
        dx = np.zeros((dy.shape[0], self.ci, self.hi, self.wi), dtype=dy.dtype)
        self.cg.deconv_gemm_nchw(self.weights, dy, dx,
                                 vpadding=self.vpadding, hpadding=self.hpadding,
                                 vstride=self.vstride, hstride=self.hstride,
                                 vdilation=self.vdilation, hdilation=self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return dx
