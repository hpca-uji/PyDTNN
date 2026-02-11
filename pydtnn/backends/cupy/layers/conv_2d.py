import cupy as np

from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.constants import DTYPE2CTYPE

class Conv2DCupy(Conv2DNumpy, LayerCupy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self._im2row = self.im2row_kernel()
        self._im2col = self.im2col_kernel()
        self._row2im = self.row2im_kernel()
        self._col2im = self.row2im_kernel()
        #----

    def im2row(self, x: np.ndarray, x_rows: np.ndarray) -> None:
        self._im2row(self.model.cuda_grid,
                    self.model.cuda_block,
                    (x, x_rows,
                     x.shape[0], self.ci, self.hi, self.wi,
                     self.kh, self.kw, self.ho, self.wo,
                     self.hpadding, self.wpadding,
                     self.hstride, self.wstride,
                     self.hdilation, self.wdilation))
    # -----

    def row2im(self, x_rows: np.ndarray, dx: np.ndarray) -> None:
        self._row2im(self.model.cuda_grid,
                    self.model.cuda_block,
                    (x_rows, dx,
                     dx.shape[0], self.ci, self.hi, self.wi,
                     self.kh, self.kw, self.ho, self.wo,
                     self.hpadding, self.wpadding,
                     self.hstride, self.wstride,
                     self.hdilation, self.wdilation))
    # -----

    def im2col(self, x: np.ndarray, x_cols: np.ndarray) -> None:
        self._im2row(self.model.cuda_grid,
                    self.model.cuda_block,
                    (x, x_cols,
                     x.shape[0], self.ci, self.hi, self.wi,
                     self.kh, self.kw, self.ho, self.wo,
                     self.hpadding, self.wpadding,
                     self.hstride, self.wstride,
                     self.hdilation, self.wdilation))
    # -----
    def col2im(self, x_cols: np.ndarray, dx: np.ndarray) -> None:
        self._col2im(self.model.cuda_grid,
                    self.model.cuda_block,
                    (x_cols, dx,
                     dx.shape[0], self.ci, self.hi, self.wi,
                     self.kh, self.kw, self.ho, self.wo,
                     self.hpadding, self.wpadding,
                     self.hstride, self.wstride,
                     self.hdilation, self.wdilation))
    # -----

####################################################################################################
####### CUDA_CODE #######
#########################

    def im2_rc_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        code = \
r"""
extern "C"
{MACROS}

__global__ void {FUNC_NAME}(const {T} *const x,
                            {T}* rows,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int hpadding,
                            int hstride, int wstride,
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

        hi = hstride * hoi + vdilation * khi - vpadding;
        wi = wstride * woi + hdilation * kwi - hpadding;

        if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            *(rows + idx) = *(x + SHIFT(ni, ci, hi, wi, c, h, w));
        else
            *(rows + idx) = ({T}) 0.0;
    }}
}}
""".format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)

        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----------------------

    def im2row_kernel(self) -> np.RawKernel:
        # x format: (N, HI, WI, CI)
        # rows format: (N, HO, WO, CI, KH, KW)
        macros=\
r"""
#define SHIFT(ni, ci, hi, wi, c, h, w) ((ni * h + hi) * w + wi) * c + ci

#define GET_N(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh * wo * ho))
#define GET_HO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh * wo)) % ho
#define GET_WO(idx, ho, wo, kh, kw, ci) (idx / (ci * kw * kh)) % wo
#define GET_CI(idx, ho, wo, kh, kw, ci) (idx / (kw * kh)) % ci
#define GET_KH(idx, ho, wo, kh, kw, ci) (idx / kw) % kh
#define GET_KW(idx, ho, wo, kh, kw, ci) (idx % kw)
"""
        self.im2_rc_kernel(func_name="im2row", macros = macros) 
    # -----

    def im2col_kernel(self) -> np.RawKernel:
        # x format: (N, CI, HI, WI)
        # cols format: (N, CI, HO, WO, KH, KW)
        macros=\
r"""
#define SHIFT(ni, ci, hi, wi, c, h, w) ((ni * c + ci) * h + hi) * w + wi

#define GET_N(idx, ho, wo, kh, kw, ci) (idx / (kw * kh * wo * ho * ci))
#define GET_HO(idx, ho, wo, kh, kw, ci) (idx / (kw * kh * wo * ho)) % ci
#define GET_WO(idx, ho, wo, kh, kw, ci) (idx / (kw * kh * wo)) % ho
#define GET_CI(idx, ho, wo, kh, kw, ci) (idx / (kw * kh)) % wo
#define GET_KH(idx, ho, wo, kh, kw, ci) (idx / kw) % kh
#define GET_KW(idx, ho, wo, kh, kw, ci) (idx % kw)
"""
        self.im2_rc_kernel(func_name="im2col", macros = macros) 
    # -----

# =======================

    def row2im_kernel(self) -> np.RawKernel:
        # dx format: (N, HI, WI, CI)
        # rows format: (N, HO, WO, CI, KH, KW)
        macros=\
r"""
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT(i, j, dim_j) (i * dim_j) + j

#define GET_N(idx, h, w, c) (idx / (c * w * h))
#define GET_H(idx, h, w, c) (idx / (c * w)) % h
#define GET_W(idx, h, w, c) (idx / c) % w
#define GET_C(idx, h, w, c) idx % c

#define GET_ROW(ni, hoi, woi, ho, wo) (ni * ho + hoi) * wo + woi
#define GET_COL(ci, khi, kwi, kh, kw) (ci * kh + khi) * kw + kwi
#define GET_COLS(c, kh, kw) c * kh * kw
"""
        self.rc2im_kernel(func_name="row2im", macros = macros) 
    # -----

    def col2im_kernel(self) -> np.RawKernel:
        # dx format: (N, CI, HI, WI)
        # cols format: (N, CI, HO, WO, KH, KW)
        macros=\
r"""
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT(i, j, dim_j) (i * dim_j) + j

#define GET_N(idx, h, w, c) (idx / (w * h * c))
#define GET_C(idx, h, w, c) (idx / (w * h)) % c
#define GET_H(idx, h, w, c) (idx / w) % h
#define GET_W(idx, h, w, c) idx % w

#define GET_ROW(ni, hoi, woi, ho, wo) (ni * ho + hoi) * wo + woi
#define GET_COL(ci, khi, kwi, kh, kw) (ci * kh + khi) * kw + kwi
#define GET_COLS(c, kh, kw) c * kh * kw
"""
        self.rc2im_kernel(func_name="col2im", macros = macros) 
    # -----

    def rc2im_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        code = \
            r"""
extern "C"
{MACROS}

__global__ void {FUNC_NAME}(const {T} *const rows,
                            {T}* dx,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int hpadding,
                            int hstride, int wstride,
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
            hoi = _hoi / hstride;
            _hoi = _hoi % hstride;

            _woi = (wi + hpadding - hdilation * kwi);
            woi = _woi / wstride;
            _woi = _woi % wstride;

            if((_hoi == 0) && (_woi == 0) && IS_BETWEEN(0, hoi, ho) && IS_BETWEEN(0, woi, wo))
            {{
                row = GET_ROW(ni, hoi, woi, ho, wo);
                col = GET_COL(ci, khi, kwi, kh, kw);
                *(dx + idx) += *(rows + SHIFT(row, col, num_cols));
            }}
        }}
    }}
}}
""".format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----------------------
