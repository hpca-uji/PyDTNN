from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.average_pool_2d import AveragePool2DNumpy
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING

from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE
if TYPE_CHECKING:
    import numpy as np


class AveragePool2DCupy(AveragePool2DNumpy, LayerCupy):

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

    def _fwd_avg_pool_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (x, y,
                       x.shape[0], self.ci, self.hi, self.wi,
                       self.kh, self.kw, self.ho, self.wo,
                       self.hpadding, self.wpadding,
                       self.hstride, self.wstride,
                       self.hdilation, self.wdilation))
    # ----

    def _fwd_avg_pool_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fwd_nhwc(self.model.cuda_grid,
                      self.model.cuda_block,
                      (x, y,
                       x.shape[0], self.ci, self.hi, self.wi,
                       self.kh, self.kw, self.ho, self.wo,
                       self.hpadding, self.wpadding,
                       self.hstride, self.wstride,
                       self.hdilation, self.wdilation))
    # ----

    def _bwd_avg_pool_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.bwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (dy, dx,
                       dy.shape[0], self.hi, self.wi, self.ci,
                       self.kh, self.kw, self.ho, self.wo,
                       self.hpadding, self.wpadding,
                       self.hstride, self.wstride,
                       self.hdilation, self.wdilation))
    # ----

    def _bwd_avg_pool_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.bwd_nhwc(self.model.cuda_grid,
                      self.model.cuda_block,
                      (dy, dx,
                       dy.shape[0], self.hi, self.wi, self.ci,
                       self.kh, self.kw, self.ho, self.wo,
                       self.hpadding, self.wpadding,
                       self.hstride, self.wstride,
                       self.hdilation, self.wdilation))
    # ----

####################################################################################################
####### CUDA_CODE #######
#########################

    def _fwd_nchw_kernel(self, macros: str) -> np.RawKernel:
        func_name = "average_pool_fwd_nchw"
        return self._fwd_kernel(func_name, macros)
    # ---

    def _fwd_nhwc_kernel(self, macros: str) -> np.RawKernel:
        func_name = "average_pool_fwd_nhwc"
        return self._fwd_kernel(func_name, macros)
    # ---

    def _fwd_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        
        # NOTE: N = n * c * ho * wo; y's format: NCHW
        code = \
            r"""
extern "C"
{MACROS}
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void {FUNC_NAME}({T}* x, {T}* y,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int hpadding, int wpadding,
                            int hstride, int wstride,
                            int hdilation, int wdilation)
{{
    int ni, ci, hoi, woi, khi, kwi, items;
    {T} accum;

    int idx;

    const int N = n * ho * wo * c;
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
        ni = GET_N(idx, n, ho, wo, c);
        hoi = GET_H(idx, n, ho, wo, c);
        woi = GET_W(idx, n, ho, wo, c);
        ci = GET_C(idx, n, ho, wo, c);

        accum = ({T}) 0.0;
        items = 0;
        
        for(khi = 0; khi < kh; khi++)
            for(kwi = 0; kwi < kw; kwi++)
        {{
            hi = wstride * hoi + hdilation * khi - hpadding;
            wi = wstride * woi + wdilation * kwi - wpadding;

            if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            {{
                accum += (*(x + SHIFT(ni, ci, hi, wi, n, c, h, w)));
                items += 1;
            }}
        }}
        *(y + idx) = ({T}) (accum / items);
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # ---


    def _bwd_nchw_kernel(self, macros: str) -> np.RawKernel:
        func_name = "average_pool_bwd_nchw"
        return self._bwd_kernel(func_name, macros)
    # ---

    def _bwd_nhwc_kernel(self, macros: str) -> np.RawKernel:
        func_name = "average_pool_bwd_nhwc"
        return self._bwd_kernel(func_name, macros)
    # ---

    def _bwd_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        
        # NOTE: N = n * ci * hi * wo; dx's format: NCHW
        code = \
            r"""
extern "C"
{MACROS}
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void {FUNC_NAME}({T}* dx, {T}* dy,
                            int n, int c, int h, int w,
                            int kh, int kw, int ho, int wo,
                            int hpadding, int wpadding,
                            int hstride, int wstride,
                            int hdilation, int wdilation)
{{
    int ni, ci, hoi, woi, khi, kwi, items;
    int idx;

    const int N_avg = n * ho * wo * c;
    const int N_pool = n * h * w * c;
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    // Getting the average

    overworkers = N_avg % num_workers;
    samples_worker = N_avg / num_workers;
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

    // NOTE: This one iterates over dy (n, c, ho, wo || n, ho, wo, c)
    for(idx = n_offset; idx < end_offset; idx++)
    {{
        // ni = GET_N(idx, n, ho, wo, c);
        hoi = GET_H(idx, n, ho, wo, c);
        woi = GET_W(idx, n, ho, wo, c);
        // ci = GET_C(idx, n, ho, wo, c);
        
        for(khi = 0; khi < kh; khi++)
            for(kwi = 0; kwi < kw; kwi++)
        {{
            hi = wstride * hoi + hdilation * khi - hpadding;
            wi = wstride * woi + wdilation * kwi - wpadding;

            if(IS_BETWEEN(0, hi, h) && IS_BETWEEN(0, wi, w))
            {{
                items += 1;
            }}
        }}
        *(dy + idx) = ({T}) ( *(dy + idx) / items);
    }}

    // Making the (average) pool

    overworkers = N_pool % num_workers;
    samples_worker = N_pool / num_workers;
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

    // NOTE: This one iterates over dy (n, c, hi, wi || n, hi, wi, c)
    for(idx = n_offset; idx < end_offset; idx++)
    {{
        ni = GET_N(idx, n, hi, wi, c);
        hi = GET_H(idx, n, hi, wi, c);
        wi = GET_W(idx, n, hi, wi, c);
        ci = GET_C(idx, n, hi, wi, c);
        
        for(khi = 0; khi < kh; khi++)
        {{
            _xx = hi + hpadding - hdilation * khi;
            xx = _xx / hstride;
            _xx %= hstride;

            for(kwi = 0; (kwi < kw) && ((_xx == 0) && IS_BETWEEN(0, xx, ho)); kwi++)
            {{
                _yy = wi + wpadding - wdilation * kwi;
                yy = _yy / wstride;
                _yy %= wstride;

                if((_yy == 0) && IS_BETWEEN(0, yy, wo))
                {{
                    *(dx + idx) += (*(dy + SHIFT(ni, ci, xx, yy, n, c, ho, wo)));
                }}
            }}
        }}
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # ---
