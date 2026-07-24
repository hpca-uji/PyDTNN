"""PyCUDA implementation of 2D Convolution layer."""

import logging
from typing import Any, override

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.pycuda.layers.abstract.conv_2d import AbstractConv2DPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.libs import cudnn as cudnn
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose

__all__ = ("Conv2DPycuda",)

logger = logging.getLogger(__name__)


class Conv2DPycuda(AbstractConv2DPycuda):
    """PyCUDA-accelerated 2D Convolution layer using cuDNN."""

    def _initializing_special_parameters(self) -> None:
        """Initialize layer-specific weight shapes based on tensor format."""
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.weights_shape = (self.co, self.ci, *self.filter_shape)
            case TensorFormat.NHWC:
                # NOTE: It is this shape, even if in the CPU version is different.
                self.weights_shape = (self.co, *self.filter_shape, self.ci)
            case _:
                raise NotImplementedError(f"{self.model.tensor_format} format not implemented.")

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        """Initialize cuDNN descriptors, algorithms, and workspace memory."""
        super()._model_init(prev_shape, x)

        # Activations y
        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # Derivative dx
        dx_gpu = gpuarray.zeros(self.x.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

        # Convolution params
        conv_mode = cudnn.cudnnConvolutionMode["CUDNN_CROSS_CORRELATION"]
        self.fwd_algo = cudnn.cudnnConvolutionFwdAlgo[
            "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
        ]
        self.bwd_dw_algo = cudnn.cudnnConvolutionBwdFilterAlgo[
            "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1"
        ]
        self.bwd_dx_algo = cudnn.cudnnConvolutionBwdDataAlgo["CUDNN_CONVOLUTION_BWD_DATA_ALGO_1"]

        # Create convolution descriptor
        self.conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(
            self.conv_desc,
            self.hpadding,
            self.wpadding,
            self.hstride,
            self.wstride,
            self.hdilation,
            self.wdilation,
            conv_mode,
            self.model.cudnn_dtype,
        )
        # Set grouping options
        # if self.grouping is Conv2D.Grouping.DEPTHWISE:
        #    cudnn.cudnnSetConvolutionGroupCount(self.conv_desc, self.ci)

        # Allow NCHW -> NHWC conversion for the use of Tensor Cores
        math_type = cudnn.cudnnMathType["CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION"]
        # math_type = cudnn.cudnnMathType['CUDNN_DEFAULT_MATH']
        # math_type = cudnn.cudnnMathType['CUDNN_TENSOR_OP_MATH']
        cudnn.cudnnSetConvolutionMathType(self.conv_desc, math_type)

        # Get output dimensions
        _, _, _ho, _wo = cudnn.cudnnGetConvolution2dForwardOutputDim(
            self.conv_desc, x.desc, self.weights.desc
        )
        assert self.ho == _ho and self.wo == _wo, "cuDNN output sizes differ from expected ones!"

        # Set to 20 the number of requested algorithms for use_cudnn_auto_conv_algo
        req_algs = 20

        self.fwd_algo = (
            cudnn.cudnnFindConvolutionForwardAlgorithm(
                self.model.cudnn_handle,
                x.desc,
                self.weights.desc,
                self.conv_desc,
                self.y.desc,
                req_algs,
            )[0].algo
            if self.model.use_cudnn_auto_conv_algo
            else cudnn.cudnnConvolutionFwdAlgo["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"]
        )

        base_conv_memory = self.model.layers[0].get_convolution_workspace_size()

        local_size = cudnn.cudnnGetConvolutionForwardWorkspaceSize(
            self.model.cudnn_handle,
            x.desc,
            self.weights.desc,
            self.conv_desc,
            self.y.desc,
            self.fwd_algo,
        )
        self.model.layers[0].check_convolution_memory(local_size)

        self.bwd_dw_algo = (
            cudnn.cudnnFindConvolutionBackwardFilterAlgorithm(
                self.model.cudnn_handle,
                x.desc,
                self.y.desc,
                self.conv_desc,
                self.weights.desc,
                req_algs,
            )[0].algo
            if self.model.use_cudnn_auto_conv_algo
            else cudnn.cudnnConvolutionBwdFilterAlgo["CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1"]
        )

        local_size = cudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(
            self.model.cudnn_handle,
            x.desc,
            self.y.desc,
            self.conv_desc,
            self.weights.desc,
            self.bwd_dw_algo,
        )
        self.model.layers[0].check_convolution_memory(local_size)

        self.bwd_dx_algo = (
            cudnn.cudnnFindConvolutionBackwardDataAlgorithm(
                self.model.cudnn_handle,
                self.weights.desc,
                self.y.desc,
                self.conv_desc,
                x.desc,
                req_algs,
            )[0].algo
            if self.model.use_cudnn_auto_conv_algo
            else cudnn.cudnnConvolutionBwdDataAlgo["CUDNN_CONVOLUTION_BWD_DATA_ALGO_1"]
        )

        local_size = cudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(
            self.model.cudnn_handle,
            self.weights.desc,
            self.y.desc,
            self.conv_desc,
            x.desc,
            self.bwd_dx_algo,
        )
        self.model.layers[0].check_convolution_memory(local_size)

        self.forward = self._forward_standard
        self.backward = self._backward_standard

        self.memory_used += self.model.layers[0].get_convolution_workspace_size() - base_conv_memory

    def _forward_standard(self, x: TensorArray) -> TensorArray:
        """Perform forward pass using cuDNN convolution."""
        alpha, beta = 1.0, 0.0
        # Compute a' = x x weights
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN
        )
        cudnn.cudnnConvolutionForward(
            self.model.cudnn_handle,
            alpha,
            x.desc,
            x.ptr_voidp,
            self.weights.desc,
            self.weights.ptr_voidp,
            self.conv_desc,
            self.fwd_algo,
            self.model.layers[0].get_convolution_workspace_ptr(),
            self.model.layers[0].get_convolution_workspace_size(),
            beta,
            self.y.desc,
            self.y.ptr_voidp,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CUDNN_SUM_BIASES,
            )
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(
                self.model.cudnn_handle,
                alpha,
                self.biases.desc,
                self.biases.ptr_voidp,
                beta,
                self.y.desc,
                self.y.ptr_voidp,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y

    def _backward_standard(self, dy: TensorArray) -> TensorArray:
        """Perform backward pass using cuDNN convolution gradients."""
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DW
        )
        # Compute dw
        cudnn.cudnnConvolutionBackwardFilter(
            self.model.cudnn_handle,
            alpha,
            self.x.desc,
            self.x.ptr_voidp,
            dy.desc,
            dy.ptr_voidp,
            self.conv_desc,
            self.bwd_dw_algo,
            self.model.layers[0].get_convolution_workspace_ptr(),
            self.model.layers[0].get_convolution_workspace_size(),
            beta,
            self.dw.desc,
            self.dw.ptr_voidp,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.use_nccl:
            # self.model.stream.synchronize()
            self.dw.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DB,
            )
            # Compute db
            cudnn.cudnnConvolutionBackwardBias(
                self.model.cudnn_handle,
                alpha,
                dy.desc,
                dy.ptr_voidp,
                beta,
                self.db.desc,
                self.db.ptr_voidp,
            )
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            # DtoH db when data parallelism and no GPU direct/NCCL is used
            if self.model.comm and not self.model.gpudirect and not self.model.use_nccl:
                # self.model.stream.synchronize()
                self.db.get_async(self.stream_2, self.db_cpu)

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_CUDNN_DX
        )
        # Compute dx
        cudnn.cudnnConvolutionBackwardData(
            self.model.cudnn_handle,
            alpha,
            self.weights.desc,
            self.weights.ptr_voidp,
            dy.desc,
            dy.ptr_voidp,
            self.conv_desc,
            self.bwd_dx_algo,
            self.model.layers[0].get_convolution_workspace_ptr(),
            self.model.layers[0].get_convolution_workspace_size(),
            beta,
            self.dx.desc,
            self.dx.ptr_voidp,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx

    @override
    def _export_weights_dw(self, key: str) -> Any:
        """Export weights or gradients to CPU, handling format transposition."""
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, kh, kw, co
                # NCHW's dst: co, ci, kh, kw
                gpu_ary = value
                cpu_ary = gpu_ary.get()
                return np.asarray(
                    format_transpose(cpu_ary, "IHWO", "OIHW"),
                    dtype=np.float64,
                    order="C",
                    copy=True,
                )
            case TensorFormat.NCHW:
                gpu_ary = value
                cpu_ary = gpu_ary.get()
                return cpu_ary
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        """Import weights or gradients from CPU, handling format transposition."""
        attribute = getattr(self, key)
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci, kh, kw
                # NHWC's dst: ci, kh, kw, co
                cpu_ary = format_transpose(value, "OIHW", "IHWO")
                attribute.set(cpu_ary)
                return
            case TensorFormat.NCHW:
                cpu_ary = value
                attribute.set(cpu_ary)
                return
            case tensor_format:
                raise TypeError(f"Unsupported tensor format ({tensor_format})")
