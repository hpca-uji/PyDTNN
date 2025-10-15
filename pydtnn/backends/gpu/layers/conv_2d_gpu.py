# noinspection PyUnresolvedReferences
from pydtnn.layers import Conv2D
from pydtnn.backends.gpu.libs import libcudnn as cudnn
# noinspection PyUnresolvedReferences
import pycuda.driver as drv
# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

from pydtnn.performance_models import *
from pydtnn.tracers import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.layer_gpu import LayerGPU
from pydtnn.backends.gpu.layers.memory_allocation import checkConvolutionMemory, getConvolutionWorkspaceSize, getConvolutionWorkspacePtr
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.utils.types import shape_t

DICT_SUPPORTED_TYPES = {np.float32: "float", np.float64: "double"}
MACROS_NCHW = \
    """
#define SHIFT_POINTER(p, c, h, w, ni, ci, hi, wi) p + ((ni * c + ci) * h + hi) * w + wi
#define SHIFT_POINTER_K(p, c, yc, ci, yci) p + (yci * c + ci)
#define INDEX_N(idx, N, n) idx * n / N
#define INDEX_C(idx, c, h, w) (idx / (h * w)) % c
#define INDEX_H(idx, c, h, w) (idx / w) % h
#define INDEX_W(idx, c, h, w) idx % w
"""
# ---

MACROS_NHWC = \
    """
#define SHIFT_POINTER(p, c, h, w, ni, ci, hi, wi) p + ((ni * h + hi) * w + wi) * c + ci
#define SHIFT_POINTER_K(p, c, yc, ci, yci) p + (ci * yc + yci)
#define INDEX_N(idx, N, n) idx * n / N
#define INDEX_H(idx, h, w, c) (idx / (w * c)) % h
#define INDEX_W(idx, h, w, c) (idx / c) % w
#define INDEX_C(idx, h, w, c) idx % c
"""
# ---


class Conv2DGPU(LayerGPU, Conv2D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fwd_algo = None
        self.fwd_time = None
        self.bwd_dw_algo = None
        self.bwd_dx_algo = None
        self.conv_desc = None
    # ---

    def initialize(self, prev_shape: shape_t, x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)

        self.stream_2 = drv.Stream()

        if self.grouping is Conv2D.Grouping.DEPTHWISE:
            self.weights_shape = (1, *self.weights_shape)
        elif self.grouping is Conv2D.Grouping.STANDARD:
            # This weight shape is required for cuDNN when NHWC is seleted!
            self.weights_shape = (self.co, *self.filter_shape, self.ci)

        self.weights_cpu = self.weights_initializer(self.weights_shape, self.model.dtype)
        weights_gpu = gpuarray.to_gpu(self.weights_cpu)
        self.weights = TensorGPU(weights_gpu, self.model.tensor_format, self.model.cudnn_dtype, TensorGPU.TensorTypeEnum.FILTER)
        # Biases
        if self.use_bias:
            self.biases_cpu = self.biases_initializer((1, self.co, 1, 1)
                                                      if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW else (1, 1, 1, self.co), self.model.dtype)
            biases_gpu = gpuarray.to_gpu(self.biases_cpu)
            self.biases = TensorGPU(biases_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.fwd_time = \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype) + \
            matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo), k=self.co,
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw, dtype=self.model.dtype)

        # Derivative dw and derivative db
        if self.model.gpudirect:
            self.dw_cpu, self.dw = TensorGPU.initialize_gpu_direct(drv, self.weights.ary.shape, self.model.dtype,
                                                                   tensor_format=self.model.tensor_format,
                                                                   cudnn_dtype=self.model.cudnn_dtype,
                                                                   gpudirect=self.model.gpudirect,
                                                                   tensor_type=TensorGPU.TensorTypeEnum.FILTER)
            if self.use_bias:
                self.db_cpu, self.db = TensorGPU.initialize_gpu_direct(self.biases.ary.shape, self.weights.ary.shape,
                                                                       self.model.dtype, tensor_format=self.model.tensor_format,
                                                                       cudnn_dtype=self.model.cudnn_dtype,
                                                                       gpudirect=self.model.gpudirect)
        else:
            self.dw_cpu, self.dw = TensorGPU.initialize_not_gpu_direct(self.weights.ary.shape, self.model.dtype,
                                                                       tensor_format=self.model.tensor_format,
                                                                       cudnn_dtype=self.model.cudnn_dtype,
                                                                       gpudirect=self.model.gpudirect,
                                                                       tensor_type=TensorGPU.TensorTypeEnum.FILTER)
            if self.use_bias:
                self.db_cpu, self.db = TensorGPU.initialize_not_gpu_direct(self.biases.ary.shape, self.model.dtype,
                                                                           tensor_format=self.model.tensor_format,
                                                                           cudnn_dtype=self.model.cudnn_dtype,
                                                                           gpudirect=self.model.gpudirect)

        match self.grouping:
            case Conv2D.Grouping.STANDARD:
                self.initialize_standard_grouping(x)
            case Conv2D.Grouping.DEPTHWISE:
                self.initialize_depthwise_grouping()
            case Conv2D.Grouping.POINTWISE:
                self.initialize_pointwise_grouping()
    # -----

    def forward(self, x: TensorGPU) -> TensorGPU:
        msg = """This is a fake forward function. It must be masked on initialization by a _forward implementation"""
        NotImplementedError(f"Conv2DGPU forward: {msg}")

    def backward(self, dy: TensorGPU) -> TensorGPU:
        msg = """This is a fake backward function. It must be masked on initialization by a _backward implementation"""
        NotImplementedError(f"Conv2DGPU backward: {msg}")

    ####################
    ## STANDARD CONV. ##
    ####################

    def initialize_standard_grouping(self, x: TensorGPU):

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
        if self.grouping is Conv2D.Grouping.DEPTHWISE:
            cudnn.cudnnSetConvolutionGroupCount(self.conv_desc, self.ci)

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

    ####################
    ###### COMMON ######
    ####################

    def cuda_sum_bias_axis_023(self, _func_name: str = "bias_sum_bwd_depthwise_conv_nchw") -> Function:
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        # np.sum(dy, axis=(0, 2, 3), out=self.db)
        code = \
            """
__global__ void {func_name}({T}* dy, {T}* db
                            int c, int h, int w,
                            int N, int num_workers)
{{
    int idx, index_c;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        index_c = (idx / (h*w)) % c;
        *(db + index_c) += *(dy + idx);
    }}
}}
"""

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ----

    def cuda_sum_bias_axis_012(self, _func_name: str = "bias_sum_bwd_depthwise_conv_nhwc") -> Function:
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        # np.sum(dy, axis=(0, 1, 2), out=self.db)
        code = \
            """
__global__ void {func_name}({T}* dy, {T}* db,
                            int c, int N,
                            int num_workers)
{{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        *(db + (idx % c)) += *(dy + idx);
    }}
}}
"""

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ----

    # ---

    ####################
    ## DEPTHWISE CONV ##
    ####################

    def cuda_depthwise_conv_2d_fwd(self, _func_name: str, _macros: str) -> Function:

        code = \
            """
{macros}
__global__ void {func_name}({T}* x, {T}* k, {T}* res,
                            int vpadding, int hpadding,
                            int vstride, int hstride,
                            int vdilation, int hdilation,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int num_workers)
{{
    int idx, cc, hi, wi, yy, xx, nn, x_x, x_y;
    int N = n * c * ho * wo;
    {T} val_k, val_x;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        cc = INDEX_C(idx, c, ho, wo);
        xx = INDEX_H(idx, c, ho, wo);
        yy = INDEX_W(idx, c, ho, wo);

        for (hi = 0; hi < kh; hi++)
        {{
            for (wi = 0; wi < kw; wi++)
            {{
                x_x = vstride * xx + vdilation * hi - vpadding;
                x_y = hstride * yy + hdilation * wi - hpadding;
                if ((0 <= x_x) && (x_x < h) && (0 <= x_y) && (x_y < w))
                {{
                    val_k = *(SHIFT_POINTER(k, c, h, w, 0, cc, hi, wi));
                    val_x = *(SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y));
                    *(SHIFT_POINTER(res, c, h, w, nn, cc, xx, yy)) += ({T}) (val_k * val_x);
                }}
            }}
        }}
    }}
}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def cuda_depthwise_conv_2d_bwd(self, _func_name: str, _macros: str) -> Function:

        code = \
            """
{macros}
__global__ void {func_name}({T}* dy, {T}* x, {T}* k,
                            {T}* dx, {T}* dw,
                            int vpadding, int hpadding,
                            int vstride, int hstride,
                            int vdilation, int hdilation,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int num_workers)
{{
    int idx, cc, khi, kwi, yy, xx, nn, x_x, x_y;
    {T} val_k, val_dy, val_x;
    int N = n * c * ho * wo;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        cc = INDEX_C(idx, c, ho, wo);
        xx = INDEX_H(idx, c, ho, wo);
        yy = INDEX_W(idx, c, ho, wo);

        val_dy = ({T}) *(SHIFT_POINTER(dy, c, h, w, nn, cc, xx, yy));
        for (khi = 0; khi < kh; khi++)
        {{
            for (kwi = 0; kwi < kw; kwi++)
            {{
                x_x = vstride * xx + vdilation * khi - vpadding;
                x_y = hstride * yy + hdilation * kwi - hpadding;
                if ((0 <= x_x) && (x_x < h) && (0 <= x_y) && (x_y < w)){{
                    val_k = *(SHIFT_POINTER(k, c, h, w, 0, cc, khi, kwi));
                    val_x = *(SHIFT_POINTER(x, c, h, w, nn, cc, x_x, x_y));
                    *(SHIFT_POINTER(dw, c, h, w, 0, cc, khi, kwi)) = ({T}) (val_x * val_dy);
                    *(SHIFT_POINTER(dx, c, h, w, nn, cc, x_x, x_y)) += ({T}) (val_k * val_dy);
                }}
            }}
        }}
    }}
}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def cuda_bias_sum_fwd_depthwise_conv(self, _func_name: str = "bias_sum_fwd_depthwise_conv") -> Function:
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        code = \
            """
__global__ void {func_name}({T}* x, {T}* bias,
                            int co, int N,
                            int num_workers)
{{
    int idx;

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        *(x + idx) += *(bias + ( idx / (N/co) ) );
    }}
}}
"""

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ----

    def initialize_depthwise_grouping(self):

        # NOTE: Seems that in PyDTNN, usually the ".x" (blockIdx.x, threadIdx.x, ...) is the only dimension used.
        self.threads = min(self.model.batch_size, 1024)
        self.blocks = max(self.model.batch_size, 1024) // self.threads + 1
        self.grid = (self.blocks, 1, 1)
        self.block = (self.threads, 1, 1)

        func_name: str = None
        macros: str = None
        self.bias_sum_bwd: Function = None

        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nchw"
                macros = MACROS_NCHW
                self.bias_sum_bwd = self.cuda_sum_bias_axis_023()
                self.forward = self._forward_depthwise_nchw
                self.backward = self._backward_depthwise_nchw
            case PYDTNN_TENSOR_FORMAT.NHWC:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nhwc"
                macros = MACROS_NHWC
                self.bias_sum_bwd = self.cuda_sum_bias_axis_012()
                self.forward = self._forward_depthwise_nhwc
                self.backward = self._backward_depthwise_nhwc
            case _:
                raise NotImplementedError(f"\"conv_2d_gpu_depthwise\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        self.fwd_func: Function = self.cuda_depthwise_conv_2d_fwd(func_name.format(fwd_bwd="fwd"), macros)
        self.bwd_func: Function = self.cuda_depthwise_conv_2d_bwd(func_name.format(fwd_bwd="bwd"), macros)
        self.bias_sum_fwd: Function = self.cuda_bias_sum_fwd_depthwise_conv()
    # ----

    def _forward_depthwise_nchw(self, x: TensorGPU) -> TensorGPU:
        self.x = x
        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        n, c, h, w = x.shape

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(x.ary, self.weights.ary, self.y.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary,
                              np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * h * w * c),
                              self.total_num_threads,
                              grid=self.grid, block=self.block,
                              stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # ----

    def _forward_depthwise_nhwc(self, x: TensorGPU) -> TensorGPU:
        self.x = x
        n, h, w, c = x.shape

        y_gpu = gpuarray.to_gpu(np.zeros(shape=(n, *self.shape), dtype=self.model.dtype))
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(x.ary, self.weights.ary, self.y.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary,
                              np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * h * w * c),
                              self.total_num_threads,
                              grid=self.grid, block=self.block,
                              stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # ----

    def _backward_depthwise_nchw(self, dy: TensorGPU) -> TensorGPU:

        n, c, h, w = dy.shape

        dx_gpu = gpuarray.zeros((n, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      self.dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads,
                      grid=self.grid, block=self.block, stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * c * h * w), self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx
    # -----

    def _backward_depthwise_nhwc(self, dy: TensorGPU) -> TensorGPU:
        n, h, w, c = dy.shape
        dx_gpu = gpuarray.zeros((n, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      self.dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads, grid=self.grid, block=self.block,
                      stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(n * h * w * c),
                              self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx
    # -----

    ####################
    ## POINTWISE CONV ##
    ####################

    def initialize_pointwise_grouping(self):

        # NOTE: Seems that in PyDTNN, usually the ".x" (blockIdx.x, threadIdx.x, ...) is the only dimension used.
        self.threads = min(self.model.batch_size, 1024)
        self.blocks = max(self.model.batch_size, 1024) // self.threads + 1
        self.grid = (self.blocks, 1, 1)
        self.block = (self.threads, 1, 1)

        func_name: str = None
        macros: str = None
        self.bias_sum_bwd: Function = None

        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nchw"
                macros = MACROS_NCHW
                self.bias_sum_bwd = self.cuda_sum_bias_axis_023()
            case PYDTNN_TENSOR_FORMAT.NHWC:
                func_name = "cuda_depthwise_conv_2d_{fwd_bwd}_nhwc"
                macros = MACROS_NHWC
                self.bias_sum_bwd = self.cuda_sum_bias_axis_012()
            case _:
                raise NotImplementedError(f"\"conv_2d_gpu_depthwise\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.total_num_threads = np.prod(self.grid, dtype=np.int32) * np.prod(self.block, dtype=np.int32)

        self.forward = self._forward_pointwise
        self.backward = self._backward_pointwise
        self.fwd_func: Function = self.cuda_depthwise_conv_2d_fwd(func_name.format(fwd_bwd="fwd"), macros)
        self.bwd_func: Function = self.cuda_depthwise_conv_2d_bwd(func_name.format(fwd_bwd="bwd"), macros)
        self.bias_sum_fwd: Function = self.cuda_bias_pointwise_conv_2d_fwd("bias_pointwise_conv_2d_fwd", macros)
    # -----

    def cuda_pointwise_conv_2d_fwd(self, _func_name: str, _macros: str) -> Function:

        code = \
            """
{macros}

__global__ void {func_name}({T}* x, {T}* k, {T}* y,
                            int n, int c, int h, int w,
                            int yc, int num_workers)
{{
    int idx, ni, ci, hi, wi, yci;
    {T} val_k, val_x;

    int N = n*c*h*w;

    // k.shape = (yc, x's c)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        val_x = *(SHIFT_POINTER(x, c, h, w, ni, ci, hi, wi));
        for(yci = 0; yci < yc; yci++)
        {{
            //y = x * k
            //val_k = k[yci][ci]; ==> val_k = k + (yci * c + ci);
            //val_k = k[ci][yci]; ==> val_k = k + (ci * kc + yci);
            val_k = *(SHIFT_POINTER_K(k, c, yc, ci, yci));
            *(SHIFT_POINTER(y, yc, h, w, ni, yci, hi, wi)) += ({T}) (val_x * val_k);
        }}
    }}
}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def cuda_pointwise_conv_2d_bwd(self, _func_name: str, _macros: str) -> Function:

        code = \
            """
{macros}

__global__ void {func_name}({T}* dy, {T}* x, {T}* k,
                            {T}* dx, {T}* dw,
                            int n, int c, int h, int w,
                            int xc, int num_workers)
{{
    int idx, ni, ci, hi, wi, xci;
    {T} val_dy, val_k, val_x;

    int N = n*c*h*w;

    // NCHW: k.shape = dw.shape = (dy's c , x's c)
    // NHWC: k.shape = dw.shape = (x's c, dy's c)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        val_dy = *SHIFT_POINTER(dy, c, h, w, ni, ci, hi, wi);
        for(xci = 0; xci < kc; xci++)
        {{
            //dw = x * dy
            val_x = *(SHIFT_POINTER(x, xc, h, w, ni, xci, hi, wi));
            *(SHIFT_POINTER_K(dw, c, xc, ci, xci)) = ({T}) (val_x * val_dy);

            //dx = w * dy
            val_k = *(SHIFT_POINTER_K(k, c, xc, ci, xci));
            *(SHIFT_POINTER(dx, kc, h, w, nn, xci, hi, wi)) += ({T}) (val_k * val_dy);
        }}
    }}
}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def cuda_bias_pointwise_conv_2d_fwd(self, _func_name: str, _macros: str) -> Function:

        code = \
            """
{macros}

__global__ void {func_name}({T}* y, {T}* b,
                            int n, int c, int h, int w,
                            int N,
                            int num_workers)
{{
    int idx, ni, ci, hi, wi;

    // self.biases.shape = (self.co,)

    for(idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += num_workers)
    {{
        ni = INDEX_N(idx, N, n);
        ci = INDEX_C(idx, c, h, w);
        hi = INDEX_H(idx, c, h, w);
        wi = INDEX_W(idx, c, h, w);

        *(SHIFT_POINTER(y, c, h, w, ni, ci, hi, wi)) += (*(b+ci));
    }}
}}
"""
        _t = DICT_SUPPORTED_TYPES[self.model.dtype]  # variable Type

        code = code.format(macros=_macros,
                           func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def _forward_pointwise(self, x: TensorGPU) -> TensorGPU:

        self.x = x
        y_gpu = gpuarray.to_gpu(np.zeros(shape=(x.shape[0], *self.shape), dtype=self.model.dtype))
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                n, c, h, w = x.shape
            case PYDTNN_TENSOR_FORMAT.NHWC:
                n, h, w, c = x.shape
            case _:
                raise NotImplementedError(f"\"Pointwise_conv_2dGPU\" is not implemented for \"{self.model.tensor_format}\" format.")

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.fwd_func(x.ary, self.weights.ary, self.y.ary,
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.co), self.total_num_threads,
                      grid=self.grid, block=self.block, stream=self.model.stream)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN_SUM_BIASES)
            self.bias_sum_fwd(x.ary, self.biases.ary,
                              np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * c * h * w),
                              self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.y
    # -----

    def _backward_pointwise(self, dy: TensorGPU) -> TensorGPU:
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                n, c, h, w = dy.shape
            case PYDTNN_TENSOR_FORMAT.NHWC:
                n, h, w, c = dy.shape
            case _:
                raise NotImplementedError(f"\"Pointwise_conv_2dGPU\" is not implemented for \"{self.model.tensor_format}\" format.")

        dx_gpu = gpuarray.zeros((n, *self.shape), self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DX)
        self.fwd_func(dy.ary, self.x.ary, self.weights.ary,
                      self.dx.ary, self.dw.ary,
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), np.int32(self.ho), np.int32(self.wo),
                      self.total_num_threads,
                      grid=self.grid, block=self.block, stream=self.model.stream)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_CUDNN_DB)
            self.bias_sum_bwd(dy.ary, self.db.ary,
                              np.int32(c), np.int32(h), np.int32(w),
                              np.int32(n * c * h * w), self.total_num_threads,
                              grid=self.grid, block=self.block, stream=self.model.stream)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return self.dx
    # -----
