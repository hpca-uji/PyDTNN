import logging
from typing import Any, override

import numpy as np

from pydtnn.backends.pycuda.layers.abstract.conv_2d import AbstractConv2DPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, PYDTNN_OPS_EVENT_enum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose

logger = logging.getLogger(__name__)


class AbstractConv2DStandardPycuda(AbstractConv2DPycuda):

    def _initializing_special_parameters(self):
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.weights_shape = (self.co, self.ci, *self.filter_shape)
            case TensorFormat.NHWC:
                self.weights_shape = (self.ci, *self.filter_shape, self.co)
                # NOTE: It is this shape, even if in the CPU version is different.
                # self.weights_shape = (self.co, *self.filter_shape, self.ci)
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")
    # -----

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        self.dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw

        self.defines_replaces["DEFINE_BIAS"] = "BIAS_DB" if self.use_bias else "DEFINE_BIAS"

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                im2_x_shape = (self.dim_c, self.dim_n)
                dw_shape = (self.co, self.dim_c)
                x_2im_var_shape = (self.dim_c, self.dim_n)

                self.im2_func = self._get_kernel(code_file_name="conv_2d_nchw", func_name="im2col")
                self._2im_func = self._get_kernel(code_file_name="conv_2d_nchw", func_name="col2im")

            case TensorFormat.NHWC:
                im2_x_shape = (self.dim_n, self.dim_c)
                dw_shape = (self.dim_c, self.co)
                x_2im_var_shape = (self.dim_n, self.dim_c)

                self.im2_func = self._get_kernel(code_file_name="conv_2d_nhwc", func_name="im2row")
                self._2im_func = self._get_kernel(code_file_name="conv_2d_nhwc", func_name="row2im")
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")

        self.im2_x = TensorArray.new_zeros(im2_x_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.x_2im_var = TensorArray.new_zeros(x_2im_var_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)

        self.y = TensorArray.new_zeros((self.model.batch_size, *self.shape), self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.dw = TensorArray.new_zeros(dw_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.dx = TensorArray.new_zeros(self.x.ary.shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
    # -----

    @override
    def _export_weights_dw(self, key: str) -> Any:
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, kh, kw, co
                # NCHW's dst: co, ci, kh, kw
                gpu_ary = value
                cpu_ary = gpu_ary.get()
                return np.asarray(format_transpose(cpu_ary, "IHWO", "OIHW"), dtype=np.float64, order="C").copy()
            case _:
                return super()._export_prop(key)
    # ------

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci, kh, kw
                # NHWC's dst: ci, kh, kw, co
                cpu_ary = np.asarray(format_transpose(value, "OIHW", "IHWO"), dtype=self.model.dtype, order="C")
                attribute.set(cpu_ary)
                return
            case _:
                return super()._import_prop(key, value)
    # ---

    def forward(self, x: TensorArray) -> TensorArray:
        # im2col / im2row
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.im2_func(x.ary, self.weights.ary,
                      self.im2_x.ary, self.y.ary,
                      self.biases.ary,
                      np.int32(self.dim_c), np.int32(self.dim_n),
                      np.int32(self.model.batch_size), np.int32(self.ci), np.int32(self.hi), np.int32(self.wi),
                      np.int32(self.co), np.int32(self.ho), np.int32(self.wo),
                      np.int32(self.kh), np.int32(self.kw),
                      np.int32(self.hpadding), np.int32(self.wpadding),
                      np.int32(self.hstride), np.int32(self.wstride),
                      np.int32(self.hdilation), np.int32(self.wdilation),
                      grid=self.grid, block=self.block,
                      stream=self.model.stream
                      )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y
    # ---

    def backward(self, dy: TensorArray) -> TensorArray:

        self.dx.fill(0)
        # im2col / im2row
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self._2im_func(dy.ary,
                       self.im2_x.ary,
                       self.weights.ary,
                       self.dw.ary,
                       self.db.ary,
                       self.dx.ary,
                       self.x_2im_var.ary,
                       np.int32(self.dim_c), np.int32(self.dim_n),
                       np.int32(self.model.batch_size), np.int32(self.ci), np.int32(self.hi), np.int32(self.wi),
                       np.int32(self.co), np.int32(self.ho), np.int32(self.wo),
                       np.int32(self.kh), np.int32(self.kw),
                       np.int32(self.hpadding), np.int32(self.wpadding),
                       np.int32(self.hstride), np.int32(self.wstride),
                       np.int32(self.hdilation), np.int32(self.wdilation),
                       grid=self.grid, block=self.block,
                       stream=self.model.stream
                       )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
    # ---
