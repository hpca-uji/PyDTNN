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
                self.weights_shape = (self.ci, *self.filter_shape, self.co)
                # NOTE: It is this shape, even if in the CPU version is different.
                #self.weights_shape = (self.co, *self.filter_shape, self.ci)
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
    # -----

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                im2_x_shape = (self.dim_c, self.dim_n)
                dw_shape = (self.co, self.dim_c)
                x_2im_var_shape = (self.dim_c, self.dim_n)

                self.im2_func = self.fwd_nchw(self.use_bias)
                self._2im_func = self._backward_nchw(self.use_bias)
            case TensorFormat.NHWC:
                im2_x_shape = (self.dim_n, self.dim_c)
                dw_shape = (self.dim_c, self.co)
                x_2im_var_shape = (self.dim_n, self.dim_c)

                self.im2_func = self.fwd_nhwc(self.use_bias)
                self._2im_func = self._backward_nhwc(self.use_bias)
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")

        self.im2_x = TensorGPU.create_zeros_tensor(im2_x_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.x_2im_var = TensorGPU.create_zeros_tensor(x_2im_var_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        
        self.y = TensorGPU.create_zeros_tensor((self.model.batch_size, *self.shape), self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.dw = TensorGPU.create_zeros_tensor(dw_shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
        self.dx = TensorGPU.create_zeros_tensor(self.x.ary.shape, self.model.dtype, self.model.tensor_format, self.model.cudnn_dtype)
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
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_CUDNN)
        self.im2_func(x.ary, self.weights.ary, 
                      self.im2_x.ary, self.y,
                      self.biases.ary,
                      np.int32(self.dim_c), np.int32(self.dim_n),
                      np.int32(self.model.batch_size), np.int32(self.ci), np.int32(self.hi), np.int32(self.wi),
                      np.int32(self.co), np.int32(self.ho), np.int32(self.wo),
                      np.int32(self.kh), np.int32(self.kw),
                      np.int32(self.vpadding), np.int32(self.hpadding),
                      np.int32(self.vstride), np.int32(self.hstride),
                      np.int32(self.vdilation), np.int32(self.hdilation),
                      grid = self.grid, block = self.block,
                      stream = self.model.stream
                    )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.y
    # ---

    def backward(self, dy: TensorGPU) -> TensorGPU:
        
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
                       np.int32(self.vpadding), np.int32(self.hpadding),
                       np.int32(self.vstride), np.int32(self.hstride),
                       np.int32(self.vdilation), np.int32(self.hdilation),
                       grid = self.grid, block = self.block,
                       stream = self.model.stream
                    )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return self.dx
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

    DB = \
"""
    for (ci = idx; ci < c; ci += num_workers)
    {{
        *(db + ci) = 0;
        for (ni = 0; ni < n; ni++)
            for (hi = 0; hi < h; hi++)
                for (wi = 0; wi < w; wi++)
        {{
            *(db + ci) += *(dy + SHIFT_DY(ni, ci, hi, wi, c, h, w));
        }}
    }}
"""
    # -- END DB --

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
#define GET_I(idx, dim_j) * idx / dim_j
#define GET_J(idx, dim_j) * idx % dim_j

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
    
    // im2col const
    const int N = c * kh * kw;
    const int dim_cols = n * self.ho * self.wo;
    // matmul const
    const int N_MATMUL = co * dim_n;
    
    // im2col vars
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
    for(i_j = idx; i_j < N_MATMUL; i_j += num_workers)
    {{
        i = GET_I(i_j, dim_n);
        j = GET_J(i_j, dim_n);
        for(k = 0; k < dim_c; k++)
        {{
            // y[i, j] += weights[i, k] * im2_var[k, j]
            *(y + SHIFT(j, i, co)) += (*(weights + SHIFT(i, k, dim_c))) * (*(im2_var + SHIFT(k, j, dim_n)));
        }}
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
                           BIAS=self.BIAS
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
#define GET_I(idx, dim_j) * idx / dim_j
#define GET_J(idx, dim_j) * idx % dim_j

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
    int ci, ki, kj, ni, hoi, hi, wi, woi, idx, row, col;
    int i, j, k, i_j;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * ho * wo;
    const int dim_cols = n * self.ho * self.wo;

    const int N_matmul = dim_n * co;

    // Im2Row
    for(row = idx; row < N; row += num_workers)
    {{  
        ni = GET_NI(row, n, ho, wo);
        hoi = GET_HO(row, n, ho, wo);
        woi = GET_WO(row, n, ho, wo);
        for (ci = 0; ci < c; ci++)
        {{
            for (ki = 0; ki < kh; ki++)
            {{
                hi = vstride * hoi + vdilation * ki - vpadding;
                for (kj = 0; kj < kw; kj++)
                {{
                    wi = hstride * woi + hdilation * kj - hpadding;
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
    // im_var = (i, k)
    // w_rows = (k, j)
    // y = (i, j)

    // im2_var.shape = (dim_n, dim_c); weights.shape "=" (dim_c, co); y.shape "=" (dim_n * co) || "=": because it's not equal, but "equivalent" in this situation.    
    for(i_j = idx; i_j < N_matmul; i_j += num_workers)
        for(k = 0; k < dim_c; k++)
    {{  
        i = GET_I(i_j, co);
        j = GET_J(i_j, co);
        // y[i, j] += im2_var[i, k] * weights[k, j]
        *(y + i_j) += (*(im2_var + SHIFT(i, k, dim_c))) * (*(weights + SHIFT(k, j, co)));
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
                            BIAS=self.BIAS
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
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

#define SHIFT_DY(ni, ci, hi, wi, n, c, h, w) (((((ni * c) + ci) * h) + hi) * w + wi)

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j
#define GET_I(idx, dim_j) * idx / dim_j
#define GET_J(idx, dim_j) * idx % dim_j

// im2col-related macros
#define GET_N(idx, n, c, h, w) idx / (w * h * c)
#define GET_C(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H(idx, n, c, h, w) (idx / w) % h
#define GET_W(idx, n, c, h, w) idx % w

__global__ void {FUNC_NAME}(const {T} *const dy,
                            const {T} *const im2_var,
                            const {T} *const weights,
                            {T}* dw, {T}* db, {T}* dx
                            {T}* col_2im_var,
                            int dim_c, int dim_n,
                            int n, int c, int h, int w,
                            int co, int ho, int wo,
                            int kh, int kw, 
                            int vpadding, int hpadding,
                            int vstride, int hstride, 
                            int vdilation, int hdilation)
{{
    // NOTE: c, h, w are the input ones and co, ho, wo are the output ones (they may differ)
    // base_dy.shape = (n, co, ho, wo)
    // im2_var.shape = (dim_n, co) || dim_n = (n * self.ho * self.wo)
    // weights.shape = (co, c, kh, kw);
    // dy.shape = (co, n, ho, wo)
    // dw.shape = (co, dim_c); dim_c = (c, kh, kw)
    // db.shape = (co, )
    // dx.shape = (n, c, h, w)
    // col_2im_var.shape = (dim_c, dim_n); dim_n = n * ho * wo; dim_c = c * kh * kw

    const int idx = blockIdx.x * blockDim.x + threadIdx.x
    const int num_workers = blockDim.x * gridDim.x;
    const int N_DW = co * c * kh * kw;
    const int N_COL2IM_VAR = dim_c * dim_n;
    const int N_COL2IM = n * c * h * w;
    const int N_TRANSPOSE = n * co * ho * wo;

    int i, j, k, i_j, dim_j, khi, kwi;
    int ni, ci, hi, wi, dy_i;

    // Matmul dy transposed and im2_var.T in and save it in dw
    // NOTE: Here dy is treated as (co, n*ho*wo); im2_var.T.shape = (n*ho*wo, ci*kh*kw)
    dim_j = N_DW / co;

    // NOTE: Remember -> dy base: NCHW, the dy needed to work: CNHW
    //dw.shape - (co, c, kh, kw)

    for(i_j = idx; i_j < N_DW; i_j += workers)
        for(k = 0; k < dim_n; k++)
    {{
        i = GET_I(i_j, dim_j);
        j = GET_J(i_j, dim_j);

        // Accessing dy like it was transposed from NCHW to CNHW
        //dy "=" (co, n, self.ho, self.wo)
        //i = "co" = GET_N(k, c, n, h, w).
        ni = GET_C(k, co, n, ho, wo);
        hi = GET_H(k, co, n, ho, wo);
        wi = GET_W(k, co, n, ho, wo);

        *(dw + i_j) += (*(dy + SHIFT_DY(ni, i, hi, wi, co, ho, wo))) * (*(im2_var + SHIFT(k, j, dim_j)));
    }}

    // np.sum(dy, axis=(0,2,3), out=db)
#if {USE_BIAS}
    {DB}
#endif

    // col_2im_var "=" (dim_c, dim_n) = (c * kh * kw, n * ho * wo)
    // mamtul(weights.reshape(co, -1).T, tranposed dy) ==>
    // mamtul(weights.reshape(co, c * kh * kw).T, tranposed dy) 
    // tranposed dy.shape = (co, n*ho*wo)
    for(i_j = idx; i_j < N_COL2IM_VAR; i_j += num_workers)
        for(k = 0; k < co; k++)
    {{
        i = GET_I(i_j, dim_n);
        j = GET_J(i_j, dim_n);

        // Accessing dy like it was transposed from NCHW to CNHW
        //dy "=" (co, n, self.ho, self.wo)
        dy_i = SHIFT(k, j, dim_n)
        ni = GET_C(dy_i, co, n, ho, wo);
        hi = GET_H(dy_i, co, n, ho, wo);
        wi = GET_W(dy_i, co, n, ho, wo);

        //col_2im_var[i][j] = weights[i][k] * dy[k][j]
        *(col_2im_var + i_j) =  (*(weights + SHIFT(i, k, co))) * (*(dy + SHIFT_DY(ni, k, hi, wi, co, ho, wo)));
    }}

    __syncthreads();

    // Col2Im
    for (i = idx; i < N_COL2IM; i += num_workers)
    {{
        ni = GET_N(i, n, c, h, w);
        ci = GET_C(i, n, c, h, w);
        hx = GET_H(i, n, c, h, w);
        wx = GET_W(i, n, c, h, w);

        for (khi = 0; khi < kh; khi++) 
            for (kwi = 0; kwi < kw; kwi++)
        {{
            // hx = vstride * xx + vdilation * khi - vpadding;
            xx = (hx + vpadding - vdilation * khi) / vstride;
            // wx = hstride * yy + hdilation * kwi - hpadding;
            yy = (wx + hpadding - hdilation * kwi) / hstride;

            x_o = (int) xx;
            y_o = (int) yy;
            
            // if (the variables have no decimals) and (are bewteen 0 and ho/wo):
            if ((x_o == xx) && (y_o == yy) && IS_BETWEEN(0, xx, ho) && IS_BETWEEN(0, yy, wo))
            {{
                row = cc * kh * kw + ii * kw + jj;
                col = nn * ho * wo + x_o * wo + y_o;
                //dx[nn, cc, x_x, x_y] += cols[row, col]
                *(dx + i) += (*(cols + SHIFT(row, col, dim_n)));
            }}
        }}
    }}

}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        func_name = "col2im_bwd_gpu"
        code = code.format(FUNC_NAME=func_name,
                            T=_t,
                            USE_BIAS=use_bias, 
                            DB = self.DB
                            )
        module = SourceModule(code).get_function(func_name)

        return module
    # -------------------------

    def _backward_nhwc(self, use_bias:bool) -> Function:
        code = \
"""
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT_DY(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

// matmul-related macros
#define SHIFT(i, j, dim_j) i * dim_j + j
#define GET_I(idx, dim_j) * idx / dim_j
#define GET_J(idx, dim_j) * idx % dim_j

// im2col-related macros
#define GET_N(idx, n, c, h, w) idx / (c * w * h)
#define GET_H(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W(idx, n, c, h, w) (idx / c) % w
#define GET_C(idx, n, c, h, w) idx % c

__global__ void {FUNC_NAME}(const {T} *const dy,
                            const {T} *const im2_var,
                            const {T} *const weights,
                            {T}* dw, {T}* db, {T}* dx
                            {T}* row_2im_var,
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
    const int N_DW = co * c * kh * kw;
    const int N_ROW2IM_VAR = dim_c * dim_n;
    const int N_ROW2IM = n * c * h * w;

    int i, j, k, i_j, dim_j, dim_k, khi, kwi;

    // NOTE: c, h, w are the input ones and co, ho, wo are the output ones (they may differ)
    // base_dy.shape = (n, co, ho, wo)
    // im2_var.shape = (dim_n, dim_c) || dim_n = (n * self.ho * self.wo)
    // weights.shape = (co, kh, kw, c)
    // dy.shape = (n, ho, wo, co)
    // dw.shape = (dim_c, co) || dim_c = (c, kh, kw)
    // db.shape = (co, )
    // dx.shape = (n, c, h, w)
    // row_2im_var.shape = (dim_c, dim_n) || dim_n = n * ho * wo; dim_c = c * kh * kw
    
    // dw = np.matmul(im2_var.T, dy.reshape(n*ho*wo, self.co)); im2_var.T.shape = (ci*kh*kw, n*ho*wo)
    for(i_j = idx; i_j < N_DW; i_j += workers)
        for(k = 0; k < dim_n; k++)
    {{
        i = GET_I(i_j, co);
        j = GET_J(i_j, co);
        *(dw + i_j) += (*(im2_var + SHIFT(k, i, dim_c))) * (*(dy + SHIFT(k, j, co)));
    }}

    // np.sum(dy, axis=(0,1,2), out=db)
#if {USE_BIAS}
    {DB}
#endif

    // row_2im_var "=" (dim_c, dim_n)
    //mamtul(weights.reshape(self.ci * self.kh * self.kw, co), tranposed dy) <== mamtul(weights.reshape(co, -1).T, tranposed dy)
    // tranposed dy.shape = (co, n*ho*wo)
    for(i_j = idx; i_j < N_ROW2IM_VAR; i_j += num_workers)
        for(k = 0; k < co; k++)
    {{
        i = GET_I(i_j, dim_n);
        j = GET_J(i_j, dim_n);

        //row_2im_var[i][j] += dy[i][k] * weights[k][j]; (weights= weights.reshape((-1, co)).T)
        *(row_2im_var + i_j) += (*(dy + SHIFT(i, k, co))) * (*(weights + SHIFT(j, k, co)));
    }}

    __syncthreads();

    // Row2Im
    for (i = idx; i < N_ROW2IM; i += num_workers)
    {{
        ni = GET_N(i, n, c, h, w);
        ci = GET_C(i, n, c, h, w);
        hx = GET_H(i, n, c, h, w);
        wx = GET_W(i, n, c, h, w);

        for (khi = 0; khi < kh; khi++) 
            for (kwi = 0; kwi < kw; kwi++)
        {{
            // hx = vstride * xx + vdilation * khi - vpadding;
            xx = (hx + vpadding - vdilation * khi) / vstride;
            // wx = hstride * yy + hdilation * kwi - hpadding;
            yy = (wx + hpadding - hdilation * kwi) / hstride;

            x_o = (int) xx;
            y_o = (int) yy;
            
            // if (the variables have no decimals) and (are bewteen 0 and ho/wo):
            if ((x_o == xx) && (y_o == yy) && IS_BETWEEN(0, xx, ho) && IS_BETWEEN(0, yy, wo))
            {{
                row = nn * ho * wo + x_o * wo + y_o;
                col = cc * kh * kw + ii * kw + jj;
                //dx[nn, x_x, x_y, cc] += rows[row, col]
                *(dx + i) += (*(row_2im_var + SHIFT(row, col, dim_c)));
            }}
        }}
    }}

}}
"""
        _t = DTYPE2CTYPE[self.model.dtype]  # variable Type

        func_name = "row2im_fwd_gpu"
        code = code.format(FUNC_NAME=func_name,
                            T=_t,
                            USE_BIAS=use_bias,
                            DB=self.DB
                            )
        module = SourceModule(code).get_function(func_name)

        return module
    # -------------------------

#########################################################################################################
