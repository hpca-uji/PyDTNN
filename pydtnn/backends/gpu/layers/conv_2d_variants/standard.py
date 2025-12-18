import numpy as np

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.conv_2d import Conv2DGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


from pydtnn.utils.tensor import TensorFormat, format_transpose
from typing import Any, override

from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.memory_allocation import checkConvolutionMemory, getConvolutionWorkspaceSize, getConvolutionWorkspacePtr
import pycuda.gpuarray as gpuarray  #type: ignore

class Conv2DStandardGPU(Conv2DGPU):

    def _initializing_special_parameters(self):
         match self.model.tensor_format:
                case TensorFormat.NCHW:
                    self.weights_shape = (self.co, self.ci, *self.filter_shape)
                case TensorFormat.NHWC:
                    # NOTE: It is this shape, even if in the CPU version is different.
                    self.weights_shape = (self.co, *self.filter_shape, self.ci)
                case _:
                    raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
    # ---

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        # Activations y
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        # Derivative dx
        dx_gpu = gpuarray.empty(self.x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Convolution params
        conv_mode = cudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
        self.fwd_algo = cudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']
        self.bwd_dw_algo = cudnn.cudnnConvolutionBwdFilterAlgo['CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']
        self.bwd_dx_algo = cudnn.cudnnConvolutionBwdDataAlgo['CUDNN_CONVOLUTION_BWD_DATA_ALGO_1']

        # Create convolution descriptor
        self.conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
        cudnn.cudnnSetConvolution2dDescriptor(self.conv_desc, self.vpadding, self.hpadding,
                                              self.vstride, self.hstride, self.vdilation, self.hdilation,
                                              conv_mode, self.model.cudnn_dtype)
        # Set grouping options
        #if self.grouping is Conv2D.Grouping.DEPTHWISE:
        #    cudnn.cudnnSetConvolutionGroupCount(self.conv_desc, self.ci)

        # Allow NCHW -> NHWC conversion for the use of Tensor Cores
        math_type = cudnn.cudnnMathType['CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION']
        # math_type = cudnn.cudnnMathType['CUDNN_DEFAULT_MATH']
        # math_type = cudnn.cudnnMathType['CUDNN_TENSOR_OP_MATH']
        cudnn.cudnnSetConvolutionMathType(self.conv_desc, math_type)

        # Get output dimensions
        _, _, _ho, _wo = cudnn.cudnnGetConvolution2dForwardOutputDim(self.conv_desc,
                                                                     x.desc, self.weights.desc)
        assert self.ho == _ho and self.wo == _wo, "cuDNN output sizes differ from expected ones!"

        # Set to 20 the number of requested algorithms for enable_cudnn_auto_conv_alg
        req_algs = 20

        self.fwd_algo = cudnn.cudnnFindConvolutionForwardAlgorithm(self.model.cudnn_handle,
                                                                   x.desc, self.weights.desc, self.conv_desc,
                                                                   self.y.desc, req_algs)[0].algo \
            if self.model.enable_cudnn_auto_conv_alg else \
            cudnn.cudnnConvolutionFwdAlgo['CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM']

        local_size = cudnn.cudnnGetConvolutionForwardWorkspaceSize(self.model.cudnn_handle,
                                                                   x.desc, self.weights.desc, self.conv_desc,
                                                                   self.y.desc, self.fwd_algo)
        checkConvolutionMemory(local_size)

        self.bwd_dw_algo = cudnn.cudnnFindConvolutionBackwardFilterAlgorithm(self.model.cudnn_handle,
                                                                             x.desc, self.y.desc, self.conv_desc,
                                                                             self.weights.desc, req_algs)[0].algo \
            if self.model.enable_cudnn_auto_conv_alg else \
            cudnn.cudnnConvolutionBwdFilterAlgo['CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1']

        local_size = cudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(self.model.cudnn_handle,
                                                                          x.desc, self.y.desc, self.conv_desc,
                                                                          self.weights.desc, self.bwd_dw_algo)
        checkConvolutionMemory(local_size)

        self.bwd_dx_algo = cudnn.cudnnFindConvolutionBackwardDataAlgorithm(self.model.cudnn_handle,
                                                                           self.weights.desc, self.y.desc,
                                                                           self.conv_desc, x.desc,
                                                                           req_algs)[0].algo \
            if self.model.enable_cudnn_auto_conv_alg else \
            cudnn.cudnnConvolutionBwdDataAlgo['CUDNN_CONVOLUTION_BWD_DATA_ALGO_1']

        local_size = cudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(self.model.cudnn_handle,
                                                                        self.weights.desc, self.y.desc,
                                                                        self.conv_desc,
                                                                        x.desc, self.bwd_dx_algo)
        checkConvolutionMemory(local_size)

        self.forward = self._forward_standard
        self.backward = self._backward_standard
    # -----


    def _forward_standard(self, x: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        # Compute a' = x x weights
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        cudnn.cudnnConvolutionForward(self.model.cudnn_handle, alpha,
                                      x.desc, x.ptr,
                                      self.weights.desc, self.weights.ptr,
                                      self.conv_desc, self.fwd_algo,
                                      getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                      self.y.desc, self.y.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            alpha, beta = 1.0, 1.0
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT,
                                         self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            # Compute a = a' + biases
            cudnn.cudnnAddTensor(self.model.cudnn_handle, alpha, self.biases.desc, self.biases.ptr,
                                 beta, self.y.desc, self.y.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y
    # -----

    def _backward_standard(self, dy: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DW)
        # Compute dw
        cudnn.cudnnConvolutionBackwardFilter(self.model.cudnn_handle, alpha,
                                             self.x.desc, self.x.ptr,
                                             dy.desc, dy.ptr, self.conv_desc, self.bwd_dw_algo,
                                             getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                             self.dw.desc, self.dw.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # DtoH dw when data parallelism and no GPU direct/NCCL is used
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            self.model.stream.synchronize()
            self.dw.ary.get_async(self.stream_2, self.dw_cpu)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            # Compute db
            cudnn.cudnnConvolutionBackwardBias(self.model.cudnn_handle, alpha,
                                               dy.desc, dy.ptr, beta,
                                               self.db.desc, self.db.ptr)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

            # DtoH db when data parallelism and no GPU direct/NCCL is used
            if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
                self.model.stream.synchronize()
                self.db.ary.get_async(self.stream_2, self.db_cpu)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        # Compute dx
        cudnn.cudnnConvolutionBackwardData(self.model.cudnn_handle, alpha,
                                           self.weights.desc, self.weights.ptr,
                                           dy.desc, dy.ptr,
                                           self.conv_desc, self.bwd_dx_algo,
                                           getConvolutionWorkspacePtr(), getConvolutionWorkspaceSize(), beta,
                                           self.dx.desc, self.dx.ptr)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
    # ----

    @override
    def _export_weights_dw(self, key: str) -> Any:
        value = getattr(self, key)

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NHWC's src: ci, kh, kw, co
                # NCHW's dst: co, ci, kh, kw
                gpu_ary = value.ary
                cpu_ary = gpu_ary.get()
                return np.asarray(format_transpose(cpu_ary, "IHWO", "OIHW"), dtype=np.float64, order="C", copy=True)
            case TensorFormat.NCHW:
                gpu_ary = value.ary
                cpu_ary = gpu_ary.get()
                return cpu_ary
            case default:
                return super()._export_prop(key)
    # ------

    @override
    def _import_weights_dw(self, key: str, value: Any) -> None:
        attribute = getattr(self, key)
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                # NCHW's src: co, ci, kh, kw
                # NHWC's dst: ci, kh, kw, co
                cpu_ary = np.asarray(format_transpose(value, "OIHW", "IHWO"), dtype=self.model.dtype, order="C", copy=None)
                attribute.ary.set(cpu_ary)
                return
            case default:
                return super()._import_prop(key, value)
    # ---
