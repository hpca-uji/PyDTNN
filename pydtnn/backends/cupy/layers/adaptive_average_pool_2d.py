from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.adaptive_average_pool_2d import AdaptiveAveragePool2DNumpy

from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING

from pydtnn.utils.constants import ArrayShape
if TYPE_CHECKING:
    import numpy as np

class AdaptiveAveragePool2DCupy(AdaptiveAveragePool2DNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        macros_nchw = \
r"""
#define GET_N(idx, n, c, h, w) (idx / (w * h * c))
#define GET_C(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H(idx, n, c, h, w) (idx / w) % h
#define GET_W(idx, n, c, h, w) (idx % w)
#define SHIFT(ni, ci, hi, wi, n, c, h, w) (((ni * c + ci) * h + hi) * w + wi)
"""

        macros_nhwc = \
r"""
#define GET_N(idx, n, c, h, w) (idx / (c * w * h))
#define GET_H(idx, n, c, h, w) (idx / (c * w)) % h
#define GET_W(idx, n, c, h, w) (idx / c) % w
#define GET_C(idx, n, c, h, w) (idx % c)
#define SHIFT(ni, ci, hi, wi, n, c, h, w) (((ni * h + hi) * w + wi) * c + ci)
"""

        self.fwd_nchw = self._fwd_nchw_kernel(macros_nchw)
        self.fwd_nhwc = self._fwd_nhwc_kernel(macros_nhwc)
        self.bwd_nchw = self._bwd_nchw_kernel(macros_nchw)
        self.bwd_nhwc = self._bwd_nhwc_kernel(macros_nhwc)
        #----

    def _fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        N = x.shape[0] * self.ci * self.ho * self.wo  # y.size
        self.fwd_nhwc(self.model.cuda_grid,
                      self.model.cuda_block,
                      (x, y,
                       x.shape[0], self.ci,
                       self.hi, self.wi,
                       self.ho, self.wo, N))
    # ----

    def _fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        N = x.shape[0] * self.ci * self.ho * self.wo  # y.size
        self.fwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (x, y,
                       x.shape[0], self.ci,
                       self.hi, self.wi,
                       self.ho, self.wo, N))
    # ----

    def _bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        N = dx.shape[0] * self.ci * self.hi * self.wi  # dx.size
        self.fwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (dx, dy,
                       dx.shape[0], self.ci,
                       self.hi, self.wi,
                       self.ho, self.wo, N))
    # ----

    def _bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        N = dx.shape[0] * self.ci * self.hi * self.wi  # dx.size
        self.fwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (dx, dy,
                       dx.shape[0], self.ci,
                       self.hi, self.wi,
                       self.ho, self.wo, N))
    # ----

####################################################################################################
####### CUDA_CODE #######
#########################

    def _fwd_nchw_kernel(self, macros: str) -> np.RawKernel:
        func_name = "adaptive_avg_pooling_fwd_nchw"
        return self._fwd_kernel(func_name, macros)
    # ---

    def _fwd_nhwc_kernel(self, macros: str) -> np.RawKernel:
        func_name = "adaptive_avg_pooling_fwd_nhwc"
        return self._fwd_kernel(func_name, macros)
    # ---

    def _fwd_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        
        # NOTE: N = n * c * ho * wo; x's format: NCHW
        code = \
            r"""
{MACROS}
#define INDEX_FIRST_ELEMENT(index, dim_in, dim_out) ((index * dim_in) / dim_out)
#define INDEX_LAST_ELEMENT(index, dim_in, dim_out) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)

__global__ void {FUNC_NAME}({T}* x, {T}* y,
                            int n, int c,
                            int hi, int wi,
                            int ho, int wo,
                            int N)
{{
    int h_start, h_end, elements_h, w_start, w_end, elements;
    int ni, ci, i, j, index_h, index_w;
    {T} add;

    int idx;
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
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
        ni = GET_N(idx, n, c, ho, wo);
        index_h = GET_H(idx, n, c, ho, wo);
        index_w = GET_W(idx, n, c, ho, wo);
        ci = GET_C(idx, n, c, ho, wo);

        h_start = INDEX_FIRST_ELEMENT(index_h, hi, ho);
        h_end = INDEX_LAST_ELEMENT(index_h, hi, ho);
        elements_h = h_end - h_start;

        w_start = INDEX_FIRST_ELEMENT(index_w, wi, wo);
        w_end = INDEX_LAST_ELEMENT(index_w, wi, wo);
        elements = elements_h * (w_end - w_start);

        add = ({T}) 0.0;
        for(i = h_start; wi < h_end; i++)
            for(j = w_start; wi < w_end; j++)
        {{
            add += *(x + SHIFT(ni, ci, i, j, n, c, hi, wi));
            *(y + SHIFT(ni, ci, index_h, index_w, n, c, ho, wo)) = add / elements;
        }}
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # ----


    def _bwd_nchw_kernel(self, macros: str) -> np.RawKernel:
        func_name = "adaptive_avg_pooling_bwd_nchw"
        return self._bwd_kernel(func_name, macros)
    # ---

    def _bwd_nhwc_kernel(self, macros: str) -> np.RawKernel:
        func_name = "adaptive_avg_pooling_bwd_nhwc"
        return self._bwd_kernel(func_name, macros)
    # ---

    def _bwd_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        
        # NOTE: N = n * ci * hi * wo; x's format: NCHW
        code = \
            r"""
{MACROS}
#define INDEX_FIRST_ELEMENT(index, dim_in, dim_out) ((index * dim_in) / dim_out)
#define INDEX_LAST_ELEMENT(index, dim_in, dim_out) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)

__global__ void {FUNC_NAME}({T}* dx, {T}* dy,
                            int n, int c,
                            int hi, int wi,
                            int ho, int wo,
                            int N)
{{
    int h_start, h_end, elements_kh, w_start, w_end, elements_kw;
    int ni, ci, index_h, index_w, index_ho, index_wo;
    {T} delta;


    // BLOCK DISTRIBUTION
    int idx;
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
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
    // BLOCK DISTRIBUTION

    for(idx = n_offset; idx < end_offset; idx++)
    {{
        ni = GET_N(idx, n, c, ho, wo);
        index_h = GET_H(idx, n, c, ho, wo);
        index_w = GET_W(idx, n, c, ho, wo);
        ci = GET_C(idx, n, c, ho, wo);

        h_start = INDEX_FIRST_ELEMENT(index_h, hi, ho);
        w_start = INDEX_FIRST_ELEMENT(index_w, wi, wo);
        h_end = INDEX_LAST_ELEMENT(index_h, hi, ho);
        w_end = INDEX_LAST_ELEMENT(index_w, wi, wo);

        for(index_ho = h_start; index_ho < h_end; index_ho++)
        {{
            elements_kh = INDEX_LAST_ELEMENT(index_h, ho, hi) - INDEX_FIRST_ELEMENT(index_h, ho, hi)
            for(index_wo = w_start; index_wo < w_end; index_wo++)
            {{
                elements_kw = INDEX_LAST_ELEMENT(index_w, wo, wi) - INDEX_FIRST_ELEMENT(index_w, wo, wi)
                delta = ({T}) (*(dy + SHIFT(idx, ci, index_ho, index_wo, n, c, ho, wo)) / (elements_kh * elements_kw));
                *(dx + SHIFT(idx, ci, index_h, index_w, n, c, hi, wi)) += delta;
            }}
        }}
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # ----
