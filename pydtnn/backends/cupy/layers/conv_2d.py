import cupy as np
from pydtnn.backends.cupy.layers.abstract.conv_2d_standard import Conv2DStandardCUPY

from pydtnn.tracers.events import PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT_enum

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, format_transpose


class Conv2DCUPY(Conv2DStandardCUPY):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super().initialize(prev_shape, x)

        # self.dim_n: Dimension where the "n" of NCHW/NHWC is used in the calculations.
        # self.dim_c: Dimension where the "c" of NCHW/NHWC is used in the calculations.
        self.dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw

        self.im2row = self.im2row_kernel()
        self.row2im = self.row2im_kernel()
        # self.im2row = self.im2row_naive
        # self.row2im = self.row2im_naive

        LIMIT_THREADS_AND_BLOCKS = 1024
        self.cuda_threads = min(self.model.batch_size, LIMIT_THREADS_AND_BLOCKS)
        self.cuda_blocks = (max(self.model.batch_size, LIMIT_THREADS_AND_BLOCKS) // self.cuda_threads) + 1
        # NOTE: Seems that in PyDTNN, usually the ".x" (blockIdx.x, threadIdx.x, ...) is the only dimension used.
        self.grid = (self.cuda_blocks, 1, 1)
        self.block = (self.cuda_threads, 1, 1)

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self.forward = self._forward_i2c_nchw
                self.backward = self._backward_i2c_nchw
                self._x_cols = np.zeros(shape=(self.dim_c, self.dim_n), dtype=self.model.dtype)

                _dw_shape = (self.co, self.dim_c)
                res_bw_shape = (self.dim_c, self.dim_n)
                dx_shape = (self.model.batch_size, self.ci, self.hi, self.wi)
            case TensorFormat.NHWC:
                self.forward = self._forward_i2c_nhwc
                self.backward = self._backward_i2c_nhwc
                self._x_rows = np.zeros(shape=(self.dim_n, self.dim_c), dtype=self.model.dtype)

                _dw_shape = (self.dim_c, self.co)
                res_bw_shape = (self.dim_n, self.dim_c)
                dx_shape = (self.model.batch_size, self.hi, self.wi, self.ci)
            case _:
                _dw_shape = (None, )
                res_bw_shape = (None, )
                dx_shape = (None,)

                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        # -

        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.dx = np.zeros(shape=dx_shape, dtype=self.model.dtype)
        self.res = np.zeros(shape=(self.dim_n, self.co), dtype=self.model.dtype)
        self._dw = np.zeros(shape=_dw_shape, dtype=self.model.dtype)
        self.res_bw = np.zeros(shape=res_bw_shape, dtype=self.model.dtype)
    # ---

    def _forward_i2c_nhwc(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        n = x.shape[0]
        dim_n = n * self.ho * self.wo
        # x_rows = np.zeros(shape=(dim_n, self.dim_c), dtype=self.model.dtype)
        x_rows = np.asarray(self._x_rows[:dim_n, :], dtype=self.model.dtype)
        res = self.res[:dim_n, :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        self.im2row(self.grid,
                    self.block,
                    (x, x_rows,
                     n, self.ci, self.hi, self.wi,
                     self.kh, self.kw, self.ho, self.wo,
                     self.hpadding, self.wpadding,
                     self.hstride, self.wstride,
                     self.hdilation, self.wdilation))
        # self.im2row (x, x_rows,
        #    n, self.ci, self.hi, self.wi,
        #    self.kh, self.kw, self.ho, self.wo,
        #    self.vpadding, self.hpadding,
        #    self.vstride, self.hstride,
        #    self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.x_rows = x_rows

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_W)
        w_cols = self.weights.reshape((-1, self.co))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        np.matmul(x_rows, w_cols, out=res,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            np.add(res, self.biases.reshape((-1, self.co)), out=res,
                   dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y = res.reshape((-1, self.ho, self.wo, self.co))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _forward_i2c_nchw(self, x: np.ndarray) -> np.ndarray:
        """Version of the forward function that uses im2col and matmul"""

        dim_n = x.shape[0] * self.ho * self.wo
        # x_cols = np.zeros(shape=(self.dim_c, dim_n), dtype=self.model.dtype)
        x_cols = np.asarray(self._x_cols[:, :dim_n], dtype=self.model.dtype)
        res = self.res[:dim_n, :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_IM2COL)
        im2col_nchw_cython(x, x_cols,
                           self.kh, self.kw, self.ho, self.wo,
                           self.hpadding, self.wpadding,
                           self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.x_cols = x_cols

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_W)
        w_rows = self.weights.reshape((self.co, -1))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_MATMUL)
        np.matmul(w_rows, x_cols, out=res.T,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_SUM_BIASES)
            np.add(res, self.biases.reshape((-1, self.co)), out=res,
                   dtype=self.model.dtype)

            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.FORWARD_RESHAPE_Y)
        y: np.ndarray = format_transpose(res.reshape((-1, self.ho, self.wo, self.co)), "NHWC", "NCHW")
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def _backward_i2c_nhwc(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses im2col and matmul"""

        n = dy.shape[0]
        res = np.asarray(self.res_bw[:(n * self.ho * self.wo), :], dtype=self.model.dtype)

        dx = self.dx[:n, :]
        dx.fill(0)  # NOTE: It is necessary that dx is filled with 0s.

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_cols: np.ndarray = dy.reshape((-1, self.co))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Weigths gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(self.x_rows.T, dy_cols, out=self._dw,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = self._dw.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Biases gradient
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 1, 2), out=self.db)
            # np.sum(dy.reshape((self.co, -1)), axis=1, out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Data gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w_rows = self.weights.reshape((-1, self.co)).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(dy_cols, w_rows, out=res,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        self.row2im(self.grid,
                    self.block,
                    (res, dx,
                     n, self.ci, self.hi, self.wi,
                     self.kh, self.kw, self.ho, self.wo,
                     self.hpadding, self.wpadding,
                     self.hstride, self.wstride,
                     self.hdilation, self.wdilation))
        # self.row2im(res, dx,
        #    n, self.ci, self.hi, self.wi,
        #    self.kh, self.kw, self.ho, self.wo,
        #    self.vpadding, self.hpadding,
        #    self.vstride, self.hstride,
        #    self.vdilation, self.hdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")

    def _backward_i2c_nchw(self, dy: np.ndarray) -> np.ndarray:
        """Version of the backward function that uses im2col and matmul"""
        res = np.asarray(self.res_bw[:, :(dy.shape[0] * self.ho * self.wo)], dtype=self.model.dtype)

        dx = self.dx[:dy.shape[0], :]
        dx.fill(0)  # NOTE: It is necessary that dx is filled with 0s.

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_DY)
        dy_rows: np.ndarray = format_transpose(dy, "NCHW", "CNHW").reshape((self.co, -1))
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Weigths gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DW_MATMUL)
        np.matmul(dy_rows, self.x_cols.T, out=self._dw,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_RESHAPE_DW)
        self.dw = self._dw.reshape(self.weights.shape)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Biases gradient
        if self.use_bias:
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_SUM_BIASES)
            np.sum(dy, axis=(0, 2, 3), out=self.db)
            # np.sum(dy.reshape((self.co, -1)), axis=1, out=self.db)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        # Data gradient
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.BACKWARD_TRANSPOSE_W)
        w_cols = self.weights.reshape((self.co, -1)).T
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_MATMUL)
        np.matmul(w_cols, dy_rows, out=res,
                  dtype=self.model.dtype)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + PYDTNN_OPS_EVENT_enum.COMP_DX_COL2IM)
        col2im_nchw_cython(res, dx,
                           dy.shape[0], self.ci, self.hi, self.wi,
                           self.kh, self.kw, self.ho, self.wo,
                           self.hpadding, self.wpadding,
                           self.hstride, self.wstride, self.hdilation, self.wdilation)
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(dx, dtype=self.model.dtype, order="C")

# ----
    def im2row_naive(self,
                     x: np.ndarray, rows: np.ndarray,
                     n: int, c: int, h: int, w: int,
                     kh: int, kw: int, ho: int, wo: int,
                     vpadding: int, hpadding: int,
                     vstride: int, hstride: int,
                     vdilation: int, hdilation: int) -> None:

        for ni in range(n):
            for ho_i in range(ho):
                for wo_i in range(wo):
                    row = (ni * ho + ho_i) * wo + wo_i
                    for ki in range(kh):
                        hi = vstride * ho_i + vdilation * ki - vpadding
                        for kj in range(kw):
                            wi = hstride * wo_i + hdilation * kj - hpadding
                            for ci in range(c):
                                col = (ci * kh + ki) * kw + kj
                                if (0 <= hi < h) and (0 <= wi < w):
                                    rows[row, col] = x[ni, hi, wi, ci]
                                else:
                                    rows[row, col] = 0.0
    # -----------------------

    def row2im_naive(self,
                     rows: np.ndarray,
                     x: np.ndarray,
                     n: int, c: int, h: int, w: int,
                     kh: int, kw: int, ho: int, wo: int,
                     vpadding: int, hpadding: int,
                     vstride: int, hstride: int,
                     vdilation: int, hdilation: int) -> None:

        for ni in range(n):
            for ho_i in range(ho):
                for wo_i in range(wo):
                    row = (ni * ho + ho_i) * wo + wo_i
                    for ci in range(c):
                        for ki in range(kh):
                            hi = vstride * ho_i + vdilation * ki - vpadding
                            if 0 <= hi < h:
                                for kj in range(kw):
                                    wi = hstride * wo_i + hdilation * kj - hpadding
                                    if 0 <= wi < w:
                                        col = (ci * kh + ki) * kw + kj
                                        x[ni, hi, wi, ci] += rows[row, col]
    # -----

    def im2row_kernel(self, func_name: str = "im2row") -> np.RawKernel:
        code = \
            r"""
extern "C"
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

#define GET_N(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh * wo * ho))
#define GET_HO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh * wo)) % ho
#define GET_WO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh)) % wo
#define GET_CI(idx, ho, wo, kh, kw, ci) (idx / (kw * kh)) % ci
#define GET_KH(idx, ho, wo, kh, kw, ci) (idx / kw) % kh
#define GET_KW(idx, ho, wo, kh, kw, ci) (idx % kw)

__global__ void {FUNC_NAME}(const {T} *const x,
                            {T}* rows,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int hpadding,
                            int vstride, int hstride,
                            int vdilation, int hdilation)
{{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * ho * wo * c * kh * kw;

    int idx, ni, ci, hoi, woi, khi, kwi, hi, wi;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    overworkers = N % num_workers;
    samples_worker = N / num_workers;
    samples_overworker = samples_worker + 1;

    if (base_idx < overworkers)
    {{
        n_samples = samples_overworker;
        n_offset = base_idx * n_samples;
    }}
    else
    {{
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (base_idx - overworkers);
    }}
    end_offset = n_offset + n_samples;

    for(idx = n_offset; idx < end_offset; idx++)
    {{
        ni = GET_N(idx, ho, wo, kh, kw, c);
        hoi = GET_HO(idx, ho, wo, kh, kw, c);
        woi = GET_WO(idx, ho, wo, kh, kw, c);
        ci = GET_CI(idx, ho, wo, kh, kw, c);
        khi = GET_KH(idx, ho, wo, kh, kw, c);
        kwi = GET_KW(idx, ho, wo, kh, kw, c);

        hi = vstride * hoi + vdilation * khi - vpadding;
        wi = hstride * woi + hdilation * kwi - hpadding;

        if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            *(rows + idx) = *(x + SHIFT(ni, ci, hi, wi, c, h, w));
        else
            *(rows + idx) = ({T}) 0.0;
    }}
}}
""".format(FUNC_NAME=func_name, T="float")

        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----------------------

    def row2im_kernel(self, func_name: str = "row2im") -> np.RawKernel:
        code = \
            r"""
extern "C"
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT(i, j, dim_j) (i * dim_j) + j

#define GET_N(idx, h, w, c) (idx / (c * w * h))
#define GET_H(idx, h, w, c) (idx / (c * w)) % h
#define GET_W(idx, h, w, c) (idx / c) % w
#define GET_C(idx, h, w, c) idx % c

#define GET_ROW(ni, hoi, woi, ho, wo) (ni * ho + hoi) * wo + woi
#define GET_COL(ci, khi, kwi, kh, kw) (ci * kh + khi) * kw + kwi
#define GET_COLS(c, kh, kw) c * kh * kw

__global__ void {FUNC_NAME}(const {T} *const rows,
                            {T}* dx,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int hpadding,
                            int vstride, int hstride,
                            int vdilation, int hdilation)
{{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * h * w * c;
    const int num_cols = GET_COLS(c, kh, kw);

    int idx, ni, ci, hoi, woi, khi, kwi, hi, wi, _hoi, _woi, row, col;
    int n_samples, n_offset, end_offset;
    int samples_worker, samples_overworker, overworkers;

    overworkers = N % num_workers;
    samples_worker = N / num_workers;
    samples_overworker = samples_worker + 1;

    if (base_idx < overworkers)
    {{
        n_samples = samples_overworker;
        n_offset = base_idx * n_samples;
    }}
    else
    {{
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (base_idx - overworkers);
    }}
    end_offset = n_offset + n_samples;

    for(idx = n_offset; idx < end_offset; idx++)
    {{
        ni = GET_N(idx, h, w, c);
        hi = GET_H(idx, h, w, c);
        wi = GET_W(idx, h, w, c);
        ci = GET_C(idx, h, w, c);

        for(khi = 0; khi < kh; khi++)
            for(kwi = 0; kwi < kw; kwi++)
        {{
            _hoi = (hi + vpadding - vdilation * khi);
            hoi = _hoi / vstride;
            _hoi = _hoi % vstride;

            _woi = (wi + hpadding - hdilation * kwi);
            woi = _woi / hstride;
            _woi = _woi % hstride;

            if((_hoi == 0) && (_woi == 0) && IS_BETWEEN(0, hoi, ho) && IS_BETWEEN(0, woi, wo))
            {{
                row = GET_ROW(ni, hoi, woi, ho, wo);
                col = GET_COL(ci, khi, kwi, kh, kw);
                *(dx + idx) += *(rows + SHIFT(row, col, num_cols));
            }}
        }}
    }}
}}
""".format(FUNC_NAME=func_name, T="float")
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----------------------
