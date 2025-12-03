import numpy as np

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.conv_2d import Conv2DGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape


from pydtnn.utils.tensor import TensorFormat, format_transpose
from typing import Any, override

from pydtnn.backends.gpu.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.memory_allocation import checkConvolutionMemory, getConvolutionWorkspaceSize, getConvolutionWorkspacePtr
import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.compiler import SourceModule  #type: ignore
from pycuda.driver import Function  #type: ignore

class Conv2DStandardGPU(Conv2DGPU):

    def _initializing_special_parameters(self):
         match self.model.tensor_format:
                case TensorFormat.NCHW:
                    self.weights_shape = (self.co, self.ci, *self.filter_shape)
                    self.im2_func = self._im2col()
                    self.forward = self._forward_nchw
                case TensorFormat.NHWC:
                    # NOTE: It is this shape, even if in the CPU version is different.
                    self.weights_shape = (self.co, *self.filter_shape, self.ci)
                    self.forward = self._forward_nhwc
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

        self.matmul = self._matmul()
        self.add_bias = self._add_bias()

        self.backward = self._backward_standard
    # -----

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

    def _forward_nchw(self, x: TensorGPU) -> TensorGPU:
        n, c, h, w = x.shape
        # im 2 col
        self.im2_func(x.ary, self.cols.ary, 
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), 
                      np.int32(self.ho), np.int32(self.wo),
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      grid = self.grid, block = self.block,
                      stream = self.model.stream
                    )
        # matrix prod

        w_cols = self.weights.reshape((-1, self.co))

        dim_i, dim_k = self.cols
        _, dim_j = self.w_cols
        y = self.y.reshape((dim_i, dim_j))

        self.matmul(self.cols.ary, w_cols, y,
                    np.int32(dim_i), np.int32(dim_k), np.int32(dim_j),
                    grid = self.grid, block = self.block,
                    stream = self.model.stream
        )

        # add bias

        if self.use_bias:
            self.add_bias(y, self.biases,
                          np.int32(dim_i), np.int32(dim_j),
                          grid = self.grid, block = self.block,
                          stream = self.model.stream
                          )

        # reshape
        self.y = y.reshape((n, c, self.ho, self.wo))

        return self.y
    # ---

    def _forward_nhwc(self, x: TensorGPU) -> TensorGPU:
        n, h, w, c = x.shape
        # im 2 col
        self.im2_func(x.ary, self.rows.ary, 
                      np.int32(n), np.int32(c), np.int32(h), np.int32(w),
                      np.int32(self.kh), np.int32(self.kw), 
                      np.int32(self.ho), np.int32(self.wo),
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      grid = self.grid, block = self.block,
                      stream = self.model.stream
                    )
        # matrix prod

        w_rows = self.weights.reshape((-1, self.co))

        dim_i, dim_k = self.rows
        _, dim_j = self.w_rows
        y = self.y.reshape((dim_i, dim_j))

        self.matmul(self.rows.ary, w_rows, y,
                    np.int32(dim_i), np.int32(dim_k), np.int32(dim_j),
                    grid = self.grid, block = self.block,
                    stream = self.model.stream
        )

        # add bias

        if self.use_bias:
            self.add_bias(y, self.biases,
                          np.int32(dim_i), np.int32(dim_j),
                          grid = self.grid, block = self.block,
                          stream = self.model.stream
                          )

        # reshape
        self.y = y.reshape((n, c, self.ho, self.wo))

        return self.y
    # ---

#########################################################################################################
## CUDA CODE ##
###############

    def _im2col(self, _func_name: str = "im2col_gpu") -> Function:
        # cols.shape = (self.dim_c, self.dim_n) = (self.ci * self.kh * self.kw, self.model.batch_size * self.ho * self.wo)
        code = \
    """
#define GET_CI(row, h, w) row / (w * h)
#define GET_KI(row, h, w) (row / w) % h
#define GET_KJ(row, h, w) row % w
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_COLS(row, col, dim_cols) row * dim_cols + col
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * c + ci) * h + hi) * w + wi

__global__ void {func_name}({T}* x, {T}* cols,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation)
{{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;
    const int N = c * kh * kw;
    const int dim_cols = n * self.ho * self.wo;

    int ci, ki, kj, ni, hoi, hi, wi, woi, idx, row, col;

    for(row = idx; row < N; row += num_workers)
    {{  
        ci = GET_CI(row, h, w);
        ki = GET_KI(row, h, w);
        kj = GET_KJ(row, h, w);
        for (ni = 0; ni < n; ni++) for (hoi = 0; hoi < ho; hoi++)
        {{
            hi = vstride * hoi + vdilation * ki - vpadding;
            for (woi = 0; woi < wo; woi++)
            {{
                wi = hstride * woi + hdilation * kj - hpadding;
                col = (ni * ho + hoi) * wo + woi;
                //cols[row, col] = ((0 <= hi) && (hi < h) && (0 <= wi) && (wi < w)) ? x[nn, cc, x_x, x_y] : ({T}) 0.0;
                if (IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
                    *(cols + SHIFT_COLS(row, col, dim_cols)) = *(x + SHIFT_X(n, ci, hi, wi, c, h, w));
                else
                    *(cols + SHIFT_COLS(row, col, dim_cols)) = ({T}) 0.0;
            }}
        }}
    }}
}}
    """
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def _im2row(self, _func_name: str = "im2row_gpu") -> Function:
        # cols.shape = (self.dim_n, self.dim_c) = (self.model.batch_size * self.ho * self.wo, self.ci * self.kh * self.kw)
        code = \
    """
#define GET_NI(cols, h, w) cols / (w * h)
#define GET_HO(cols, h, w) (cols / w) % h
#define GET_WO(cols, h, w) cols % w
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_ROWS(row, col, dim_cols) row * dim_cols + col
// NOTE: This is NHWC
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

__global__ void {func_name}({T}* x, {T}* rows,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation)
{{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * ho * wo;
    const int dim_cols = n * self.ho * self.wo;

    int ci, ki, kj, ni, hoi, hi, wi, woi, idx, row, col;

    for(row = idx; col < N; col += num_workers)
    {{  
        ni = GET_NI(col, n, ho, wo);
        hoi = GET_HO(col, n, ho, wo);
        woi = GET_WO(col, n, ho, wo);
        for (ki = 0; ki < kh; ki++)
        {{
            hi = vstride * hoi + vdilation * ki - vpadding;
            for (woi = 0; woi < wo; woi++)
            {{
                wi = hstride * woi + hdilation * kj - hpadding;
                for (ci = 0; ci < c; ci++)
                {{
                    col = (ni * ho + hoi) * wo + woi;
                    //rows[row, col] = ((0 <= hi) && (hi < h) && (0 <= wi) && (wi < w)) ? x[nn, cc, x_x, x_y] : ({T}) 0.0;
                    if (IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
                        *(rows + SHIFT_ROWS(row, col, dim_cols)) = *(x + SHIFT_X(n, ci, hi, wi, c, h, w));
                    else
                        *(rows + SHIFT_ROWS(row, col, dim_cols)) = ({T}) 0.0;
                }}
            }}
        }}
    }}
}}
    """
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type
        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def _matmul(self, _func_name: str = "matmul_gpu") -> Function:
        # cols.shape = (self.dim_n, self.dim_c) = (self.model.batch_size * self.ho * self.wo, self.ci * self.kh * self.kw)
        code = \
    """
#define SHIFT(i, j, dim_j) i * dim_j + j

// NOTE: It's assumed C is initialized to 0
// A=iXk, B=kXj, C=iXj
__global__ void {func_name}(const {T} *const A, 
                            const {T} *const B, 
                            {T} * const C, int dim_i, 
                            int dim_k, int dim_j)
{{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;

    int i, j, k;

    for(i = idx; i < dim_i; i += num_workers) 
        for(k = 0; j < dim_k; k++) 
            for(j = 0; j < dim_j; j++)
    {{
        // C[i, j] += A[i, k] + B[k, j]
        *(C + SHIFT(i, j, dim_j)) += (*(A + SHIFT(i, k, dim_k))) * (*(B + SHIFT(k, j, dim_j)))
    }}
}}
    """
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def _add_bias(self, _func_name: str = "add_bias_gpu") -> Function:
        # cols.shape = (self.dim_n, self.dim_c) = (self.model.batch_size * self.ho * self.wo, self.ci * self.kh * self.kw)
        code = \
    """
#define SHIFT(i, j, dim_j) i * dim_j + j

__global__ void {func_name}({T} *const A,
                            const {T} *const bias,
                            int dim_i, int dim_j)
{{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;

    int i, j;

    for(i = idx; i < dim_i; i += num_workers) 
        for(j = 0; j < dim_j; j++)
    {{
        // A[i, j] += bias[j];
        *(A + SHIFT(i, j, dim_j)) += *(bias + j);
    }}
}}
    """
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(func_name=_func_name,
                           T=_t
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---

    def _gradient_bias(self, is_nchw: bool) -> Function:
        # cols.shape = (self.dim_n, self.dim_c) = (self.model.batch_size * self.ho * self.wo, self.ci * self.kh * self.kw)
        code = \
    """
#ifdef {IS_NCHW}
    //SHIFT_NCHW
    #define SHIFT(ni, ci, hi, wi, c_dim, h_dim, w_dim) (((ni * c_dim + ci) * h_dim + hi) * w_dim + wi)
#else
    //SHIFT_NHWC
    #define SHIFT(ni, ci, hi, wi, c_dim, h_dim, w_dim) (((ni * h_dim + hi) * w_dim + wi) * c_dim + ci)

__global__ void {func_name}({T} *const dbias,
                            const {T} *const dy,
                            int n, int c, int h, int w)
{{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;

    int ni, ci, hi, wi;

    for(ci = idx; ci < c; ci += num_workers)
        for(ni = 0; ni < n; ni++)
            for(hi = 0; hi < h; hi++)
                for(wi = 0; wi < w; wi++)
    {{
        *(dbias + ci) += *(dy + SHIFT(ni, ci, hi, wi, c, h, w))
    }}
}}
    """
        _func_name = f"gradient_bias_{'nchw' if is_nchw else 'nhwc'}_gpu"
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(func_name=_func_name,
                           T=_t,
                           IS_NCHW=is_nchw
                           )
        module = SourceModule(code).get_function(_func_name)

        return module
    # ---


#########################################################################################################
