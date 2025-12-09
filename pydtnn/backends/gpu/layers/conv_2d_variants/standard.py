import numpy as np

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum
from pydtnn.backends.gpu.layers.conv_2d import Conv2DGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape


from pydtnn.utils.tensor import TensorFormat, format_transpose
from typing import Any, override

from pycuda.compiler import SourceModule  #type: ignore
from pycuda.driver import Function  #type: ignore

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

        # TODO:
        # > Hacer que "res" sea "x_cols/x_rows".
        # > Hacer que el "res" de i2c sea y.

        self.y = TensorGPU.create_zeros_tensor((self.model.batch_size, *self.shape), self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.dx = TensorGPU.create_zeros_tensor(self.x.ary.shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)

        self.dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.weights_shape = (self.co, self.ci, *self.filter_shape)
                self.weights_im2_shape = (self.co, int(np.prod((self.ci, *self.filter_shape))))
                self.im2_x_shape = (self.dim_c, self.dim_n)
                self.res_shape = (self.co, self.dim_n)
                self._dw_shape = (self.co, self.dim_c)
                self.y_shape = (self.model.batch_size, self.co, self.ho, self.wo)

                self.dim_i_fwd, self.dim_k_fwd = self.im2_x_shape
                _, self.dim_j_fwd = self.weights_im2_shape # (dim_k, dim_j)

                self.im2_func = self._im2col()
                self.backward = self._backward_standard
            case TensorFormat.NHWC:
                # NOTE: It is this shape, even if in the CPU version is different.
                self.weights_shape = (self.co, *self.filter_shape, self.ci)
                self.weights_im2_shape = (int(np.prod((self.ci, *self.filter_shape))), self.co)
                self.im2_x_shape = (self.dim_n, self.dim_c)
                self.res_shape = (self.dim_n, self.co)
                self._dw_shape = (self.dim_c, self.co)
                self.y_shape = (self.model.batch_size, self.ho, self.wo, self.co)

                self.dim_i_fwd, self.dim_k_fwd = self.weights_im2_shape
                _, self.dim_j_fwd = self.im2_x_shape # (dim_k, dim_j)
                
                self.im2_func = self._im2row()
                self.backward = self._backward_standard
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        
        self.im2_x = TensorGPU.create_zeros_tensor(self.im2_x_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.res = TensorGPU.create_zeros_tensor(self.res_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self._dw = TensorGPU.create_zeros_tensor(self._dw_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.matmul_matrices = (self.weights.ary, self.im2_x.ary)
            case TensorFormat.NHWC:
                self.matmul_matrices = (self.im2_x.ary, self.weights.ary)
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")


        self.matmul = self._matmul()
        self.add_bias = self._add_bias(self.model.tensor_format is TensorFormat.NCHW)
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

    def forward(self, x: TensorGPU) -> TensorGPU:
        # im2col / im2row
        self.im2_func(x.ary, self.im2_x.ary, 
                      np.int32(self.model.batch_size), np.int32(self.ci), 
                      np.int32(self.hi), np.int32(self.wi),
                      np.int32(self.kh), np.int32(self.kw), 
                      np.int32(self.ho), np.int32(self.wo),
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      grid = self.grid, block = self.block,
                      stream = self.model.stream
                    )
        
        # matrix prod
        # NOTE: in this case self.weights' shape considered as "(self.co, int(np.prod((self.ci, *self.filter_shape))))" in NCHW format and "(int(np.prod((self.ci, *self.filter_shape))), self.co)" in NHWC.
        #self.matmul(*self.matmul_matrices, self.y,
        self.matmul(self.weights.ary, self.im2_x.ary, self.y,
                    np.int32(self.dim_i_fwd), np.int32(self.dim_k_fwd), np.int32(self.dim_j_fwd),
                    grid = self.grid, block = self.block,
                    stream = self.model.stream
                )

        # add bias
        if self.use_bias:
            self.add_bias(self.y.ary, self.biases.ary,
                          np.int32(self.dim_i_fwd), 
                          np.int32(self.dim_j_fwd),
                          grid = self.grid, block = self.block,
                          stream = self.model.stream
                          )
        # --
        return self.y
    # ---

#########################################################################################################
## CUDA CODE ##
###############
    #========================
    #= FORWARD-related code =
    #========================

    BIAS= \
"""
    for(i = idx; i < dim_n; i += num_workers)
        for(j = 0; j < co; j++)
    {{
        *(im2_var + SHIFT(i, j, co)) += (*(bias + j));
    }}
""" 
    # -- END BIAS --


    def fwd_nchw(self, use_bias: bool) -> Function:
        # im2_var.shape = (self.dim_c, self.dim_n) = (self.ci * self.kh * self.kw, self.model.batch_size * self.ho * self.wo)
        code = \
"""
// im2col-related macros
#define GET_CI(row, h, w) row / (w * h)
#define GET_KI(row, h, w) (row / w) % h
#define GET_KJ(row, h, w) row % w
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_COLS(row, col, dim_cols) row * dim_cols + col
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * c + ci) * h + hi) * w + wi

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j

__global__ void {FUNC_NAME}(const {T} *const x,
                            const {T} *const weights,
                            {T}* im2_var, {T}* y,
                            {T}* bias,
                            int dim_c, int dim_n,
                            int n, int c, int h, int w,
                            int co, int ho, int wo,
                            int kh, int kw, 
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation)
{{  
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;
    
    // im2col vars
    const int N = c * kh * kw;
    const int dim_cols = n * self.ho * self.wo;
    int ci, ki, kj, ni, hoi, hi, wi, woi, idx, row, col;
    // matmul vars
    int i, j, k;

    // Im2Col
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
                //im2_var[row, col] = ((0 <= hi) && (hi < h) && (0 <= wi) && (wi < w)) ? x[nn, cc, x_x, x_y] : ({T}) 0.0;
                if (IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
                    *(im2_var + SHIFT_COLS(row, col, dim_cols)) = *(x + SHIFT_X(n, ci, hi, wi, c, h, w));
                else
                    *(im2_var + SHIFT_COLS(row, col, dim_cols)) = ({T}) 0.0;
            }}
        }}
    }}
    __syncthreads();

    // Matmul - w_rows X x_cols = y.T
    // weights.shape "=" (co, dim_c); im2_var.shape = (dim_c, dim_n); y.T "="(co, dim_n); y.shape "=" (dim_n, co) || "=": because it's not equal, but "equivalent" in this situation.
    for(j = idx; j < dim_n; j += num_workers) 
        for(k = 0; k < dim_c; k++) 
            for(i = 0; i < co; i++)
    {{
        // y[j, i] += weights[i, k] * im2_var[k, j]
        *(y + SHIFT(j, i, dim_n)) += (*(weights + SHIFT(i, k, dim_c))) * (*(im2_var + SHIFT(k, j, dim_n)));
    }}
#if {USE_BIAS}

    __syncthreads();
    {BIAS}
#endif

}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        func_name = "im2col_fwd_gpu"
        code = code.format(FUNC_NAME=func_name,
                           T=_t,
                           USE_BIAS=use_bias,
                           BIAS_=self.BIAS
                           )
        module = SourceModule(code).get_function(func_name)

        return module
    # -------------------------

    def fwd_nhwc(self, use_bias:bool) -> Function:
        # cols.shape = (self.dim_n, self.dim_c) = (self.model.batch_size * self.ho * self.wo, self.ci * self.kh * self.kw)
        code = \
"""
#define GET_NI(row, h, w) row / (w * h)
#define GET_HO(row, h, w) (row / w) % h
#define GET_WO(row, h, w) row % w
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_ROWS(row, col, dim_cols) row * dim_cols + col
// NOTE: This is NHWC
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j

__global__ void {FUNC_NAME}(const {T} *const x,
                            const {T} *const weights,
                            {T}* im2_var, {T}* y,
                            {T}* bias,
                            int dim_c, int dim_n,
                            int n, int c, int h, int w,
                            int co, int ho, int wo,
                            int kh, int kw, 
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation)
{{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * ho * wo;
    const int dim_cols = n * self.ho * self.wo;

    int ci, ki, kj, ni, hoi, hi, wi, woi, idx, row, col;
    int i, j, k;

    // Im2Row
    for(row = idx; row < N; row += num_workers)
    {{  
        ni = GET_NI(row, n, ho, wo);
        hoi = GET_HO(row, n, ho, wo);
        woi = GET_WO(row, n, ho, wo);
        for (ki = 0; ki < kh; ki++)
        {{
            hi = vstride * hoi + vdilation * ki - vpadding;
            for (woi = 0; woi < wo; woi++)
            {{
                wi = hstride * woi + hdilation * kj - hpadding;
                for (ci = 0; ci < c; ci++)
                {{
                    col = (ni * ho + hoi) * wo + woi;
                    //im2_var[row, col] = ((0 <= hi) && (hi < h) && (0 <= wi) && (wi < w)) ? x[nn, cc, x_x, x_y] : ({T}) 0.0;
                    if (IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
                        *(im2_var + SHIFT_ROWS(row, col, dim_cols)) = *(x + SHIFT_X(n, ci, hi, wi, c, h, w));
                    else
                        *(im2_var + SHIFT_ROWS(row, col, dim_cols)) = ({T}) 0.0;
                }}
            }}
        }}
    }}

    __syncthreads();

    // Matmul - im2_var X w_rows = y
    im_var = (i, k)
    w_rows = (k, j)
    y = (i, j)
    // im2_var.shape = (dim_n, dim_c); weights.shape "=" (dim_c, co); y.shape "=" (dim_n, co) || "=": because it's not equal, but "equivalent" in this situation.
    for(i = idx; i < dim_n; i += num_workers) 
        for(k = 0; k < dim_c; k++) 
            for(j = 0; j < co; j++)
    {{
        // y[i, j] += im2_var[i, k] * weights[k, j]
        *(y + SHIFT(i, j, co)) += (*(im2_var + SHIFT(i, k, dim_c))) * (*(weights + SHIFT(k, j, co)));
    }}
#if {USE_BIAS}

    __syncthreads();
    {BIAS}
#endif

}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        func_name = "im2row_fwd_gpu"
        code = code.format(FUNC_NAME=func_name,
                            T=_t,
                            USE_BIAS=use_bias,
                            BIAS_=self.BIAS
                            )
        module = SourceModule(code).get_function(func_name)

        return module
    # -------------------------
    
    #=========================
    #= BACKWARD-related code =
    #=========================

    def _backward_nchw(self, use_bias:bool) -> Function:
        code = \
"""
#define GET_NI(row, h, w) row / (w * h)
#define GET_HO(row, h, w) (row / w) % h
#define GET_WO(row, h, w) row % w
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_ROWS(row, col, dim_cols) row * dim_cols + col
// NOTE: This is NHWC
#define SHIFT_X(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j

__global__ void {FUNC_NAME}(const {T} *const dy,
                            const {T} *const im2_var,
                            const {T} *const weights,
                            {T}* dw, {T}* db, {T}* dx
                            {T}* bias,
                            int dim_c, int dim_n,
                            int n, int c, int h, int w,
                            int co, int ho, int wo,
                            int kh, int kw, 
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation)
{{
    // Transpose dy from NCHW to CNHW

    // Matmul dy transposed and im2_var.T in and save it in dw
    // NOTA: ¿Podría ser más rápido si se juntan dimensiones y se saca en qué i y en que j se está?

    // NOTE: Here dy is treated as (co, n*ho*wo); im2_var.T.shape = (n*ho*wo, ci*kh*kw)
    dim_k = n*ho*wo;
    dim_j = ci*kh*kw;

    for(co_i=idx; co_i < c; co_i++)
        for(k = 0; k < dim_k; k++)
            for(j = 0; j < dim_j; j++)
    {{
        *(dw + SHIFT(co_i, j, dim_j)) += (*(dy + SHIFT(co_i, k, dim_k))) * (*(im2_var + SHIFT(k, j, dim_j)));
    }}

    // NOTE: Here dy is treated as (co, n*ho*wo); im2_var.T.shape = (n*ho*wo, ci*kh*kw)
    dim_k = n*ho*wo;
    dim_j = ci*kh*kw;
    N = co*n*ho*wo

    for(co_i=idx; co_i < c; co_i++)
        for(k = 0; k < dim_k; k++)
            for(j = 0; j < dim_j; j++)
    {{
        *(dw + SHIFT(co_i, j, dim_j)) += (*(dy + SHIFT(co_i, k, dim_k))) * (*(im2_var + SHIFT(k, j, dim_j)));
    }}

    // np.sum(dy, axis=(0,2,3), out=db)

    //mamtul(weights.reshape(co, -1).T, tranposed dy)

    // Col2Im

}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        func_name = "im2row_fwd_gpu"
        code = code.format(FUNC_NAME=func_name,
                            T=_t,
                            USE_BIAS=use_bias,
                            BIAS_=self.BIAS
                            )
        module = SourceModule(code).get_function(func_name)

        return module
    # -------------------------


    # 

    def _gradient_bias(self, is_nchw: bool) -> Function:
        # cols.shape = (self.dim_n, self.dim_c) = (self.model.batch_size * self.ho * self.wo, self.ci * self.kh * self.kw)
        code = \
"""
#if {IS_NCHW}
    //SHIFT_NCHW
    #define SHIFT(ni, ci, hi, wi, c_dim, h_dim, w_dim) (((ni * c_dim + ci) * h_dim + hi) * w_dim + wi)
#else
    //SHIFT_NHWC
    #define SHIFT(ni, ci, hi, wi, c_dim, h_dim, w_dim) (((ni * h_dim + hi) * w_dim + wi) * c_dim + ci)
#endif

__global__ void {FUNC_NAME}({T} *const dbias,
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
        func_name = f"gradient_bias_{'nchw' if is_nchw else 'nhwc'}_gpu"
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        code = code.format(FUNC_NAME=func_name,
                           T=_t,
                           IS_NCHW=is_nchw
                           )
        module = SourceModule(code).get_function(func_name)

        return module
    # ---


#########################################################################################################
