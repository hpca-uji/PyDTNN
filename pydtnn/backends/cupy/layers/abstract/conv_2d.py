from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.utils.constants import ArrayShape
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
import cupy as np
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.cython.layers.layer import LayerCython
import logging

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.layers.abstract.conv_2d import AbstractConv2D
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class AbstractConv2DCupy(AbstractConv2DNumpy, AbstractConv2D[np.ndarray], LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = np.cuda.Stream()

        self._im2row = self.im2row_kernel()
        self._im2col = self.im2col_kernel()
        self._row2im = self.row2im_kernel()
        self._col2im = self.row2im_kernel()
        # ----

    def im2row(self, x: np.ndarray, x_rows: np.ndarray) -> None:
        # return super().im2row(x, x_rows)
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
        # return super().im2row(x_rows, dx)
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
        # return super().im2col(x, x_cols)
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
        # return super().im2row(x_cols, dx)
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
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

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
        ni = GET_N(idx, n, c, ho, wo, kh, kw);
        hoi = GET_HO(idx, n, c, ho, wo, kh, kw);
        woi = GET_WO(idx, n, c, ho, wo, kh, kw);
        ci = GET_CI(idx, n, c, ho, wo, kh, kw);
        khi = GET_KH(idx, n, c, ho, wo, kh, kw);
        kwi = GET_KW(idx, n, c, ho, wo, kh, kw);

        hi = hstride * hoi + vdilation * khi - vpadding;
        wi = wstride * woi + hdilation * kwi - hpadding;

        if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            *(rows + idx) = *(x + SHIFT(ni, ci, hi, wi, n, c, h, w));
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
        macros =\
            r"""
#define SHIFT(ni, ci, hi, wi, n, c, h, w) ((ni * h + hi) * w + wi) * c + ci

#define GET_N(idx, n, ci, ho, wo, kh, kw) (idx / (ci * kw * kh * wo * ho))
#define GET_HO(idx, n, ci, ho, wo, kh, kw) (idx / (ci * kw * kh * wo)) % ho
#define GET_WO(idx, n, ci, ho, wo, kh, kw) (idx / (ci * kw * kh)) % wo
#define GET_CI(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh)) % ci
#define GET_KH(idx, n, ci, ho, wo, kh, kw) (idx / kw) % kh
#define GET_KW(idx, n, ci, ho, wo, kh, kw) (idx % kw)
"""
        return self.im2_rc_kernel(func_name="im2row", macros=macros)
    # -----

    def im2col_kernel(self) -> np.RawKernel:
        # x format: (N, CI, HI, WI)
        # cols format: (N, CI, HO, WO, KH, KW)
        macros =\
            r"""
#define SHIFT(ni, ci, hi, wi, n, c, h, w) ((ni * c + ci) * h + hi) * w + wi

#define GET_N(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh * wo * ho * ci))
#define GET_HO(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh * wo * ho)) % ci
#define GET_WO(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh * wo)) % ho
#define GET_CI(idx, n, ci, ho, wo, kh, kw) (idx / (kw * kh)) % wo
#define GET_KH(idx, n, ci, ho, wo, kh, kw) (idx / kw) % kh
#define GET_KW(idx, n, ci, ho, wo, kh, kw) (idx % kw)
"""
        return self.im2_rc_kernel(func_name="im2col", macros=macros)
    # -----

# =======================

    def row2im_kernel(self) -> np.RawKernel:
        # dx format: (N, HI, WI, CI)
        # rows format: (N, HO, WO, CI, KH, KW)
        macros =\
            r"""
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT(i, j, dim_j) (i * dim_j) + j

#define GET_N(idx, n, c, h, w) (idx / (c * w * h))
#define GET_H(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W(idx, n, c, h, w) (idx / c) % w
#define GET_C(idx, n, c, h, w) idx % c

#define GET_ROW(ni, hoi, woi, ho, wo) (ni * ho + hoi) * wo + woi
#define GET_COL(ci, khi, kwi, kh, kw) (ci * kh + khi) * kw + kwi
#define GET_COLS(c, kh, kw) c * kh * kw
"""
        return self.rc2im_kernel(func_name="row2im", macros=macros)
    # -----

    def col2im_kernel(self) -> np.RawKernel:
        # dx format: (N, CI, HI, WI)
        # cols format: (N, CI, HO, WO, KH, KW)
        macros =\
            r"""
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)
#define SHIFT(i, j, dim_j) (i * dim_j) + j

#define GET_N(idx, n, c, h, w) (idx / (w * h * c))
#define GET_C(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H(idx, n, c, h, w) (idx / w) % h
#define GET_W(idx, n, c, h, w) idx % w

#define GET_ROW(ni, hoi, woi, ho, wo) (ni * ho + hoi) * wo + woi
#define GET_COL(ci, khi, kwi, kh, kw) (ci * kh + khi) * kw + kwi
#define GET_COLS(c, kh, kw) c * kh * kw
"""
        return self.rc2im_kernel(func_name="col2im", macros=macros)
    # -----

    def rc2im_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        code = \
            r"""
extern "C"
{MACROS}
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

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
        ni = GET_N(idx, n, c, h, w);
        hi = GET_H(idx, n, c, h, w);
        wi = GET_W(idx, n, c, h, w);
        ci = GET_C(idx, n, c, h, w);

        for(khi = 0; khi < kh; khi++)
        {{
            _hoi = (hi + vpadding - vdilation * khi);
            hoi = _hoi / hstride;
            _hoi = _hoi % hstride;

            for(kwi = 0; (kwi < kw) && ((_hoi == 0) && IS_BETWEEN(0, hoi, ho)); kwi++)
            {{
                _woi = (wi + hpadding - hdilation * kwi);
                woi = _woi / wstride;
                _woi = _woi % wstride;

                if((_woi == 0) && IS_BETWEEN(0, woi, wo))
                {{
                    row = GET_ROW(ni, hoi, woi, ho, wo);
                    col = GET_COL(ci, khi, kwi, kh, kw);
                    *(dx + idx) += *(rows + SHIFT(row, col, num_cols));
                }}
            }}
        }}
    }}
}}
""".format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----------------------
