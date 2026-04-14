import logging
logger = logging.getLogger(__name__)

from pydtnn.backends.cupy.layers.layer import LayerCupy
from pydtnn.backends.numpy.layers.conv_2d_depthwise import Conv2DDepthwiseNumpy
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class Conv2DDepthwiseCython(Conv2DDepthwiseNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        macros_nchw = \
r"""
#define GET_N(idx, n, c, h, w) (idx / (w * h * c))
#define GET_C(idx, n, c, h, w) (idx / (w * h)) % c
#define GET_H(idx, n, c, h, w) (idx / w) %h
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

        self.fwd_nhwc = self._fwd_nhwc_kernel(macros_nhwc)
        self.fwd_nchw = self._fwd_nchw_kernel(macros_nchw)
        self.bwd_nhwc = self._bwd_nhwc_kernel(macros_nhwc)
        self.bwd_nchw = self._bwd_nchw_kernel(macros_nchw)
        #----

    def _conv_fwd_nhwc(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fwd_nhwc(self.model.cuda_grid,
                      self.model.cuda_block,
                      (x, self.weights, y,
                       x.shape[0], self.ci, self.hi, self.wi,
                       self.ho, self.wo, self.kh, self.kw,
                       self.hpadding, self.wpadding,
                       self.hstride, self.wstride,
                       self.hdilation, self.wdilation))
    # ----

    def _conv_fwd_nchw(self, x: np.ndarray, y: np.ndarray) -> None:
        self.fwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (x, self.weights, y,
                      x.shape[0], self.ci, self.hi, self.wi,
                      self.ho, self.wo, self.kh, self.kw,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # ----

    def _conv_bwd_nhwc(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.bwd_nhwc(self.model.cuda_grid,
                      self.model.cuda_block,
                      (dx, dy, self.x,
                      self.weights, self.dw,
                      dy.shape[0], self.ci, self.hi, self.wi,
                      self.ho, self.wo, self.kh, self.kw,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # ----

    def _conv_bwd_nchw(self, dx: np.ndarray, dy: np.ndarray) -> None:
        self.bwd_nchw(self.model.cuda_grid,
                      self.model.cuda_block,
                      (dx, dy, self.x,
                      self.weights, self.dw,
                      dy.shape[0], self.ci, self.hi, self.wi,
                      self.ho, self.wo, self.kh, self.kw,
                      self.hpadding, self.wpadding,
                      self.hstride, self.wstride,
                      self.hdilation, self.wdilation))
    # ----

####################################################################################################
####### CUDA_CODE #######
#########################

    def _fwd_nchw_kernel(self, macros: str) -> np.RawKernel:
        func_name = "depthwise_conv_fwd_nchw"
        return self._fwd_kernel(func_name, macros)
    # ---

    def _fwd_nhwc_kernel(self, macros: str) -> np.RawKernel:
        func_name = "depthwise_conv_fwd_nhwc"
        return self._fwd_kernel(func_name, macros)
    # ---

    def _fwd_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        
        # NOTE: N = n * c * ho * wo; x's format: NCHW
        code = \
            r"""
extern "C"
{MACROS}
#define SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw) ((ci * kh + khi) * kw + kwi)
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void {FUNC_NAME}({T}* x, {T}* weights, {T}* y,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int hpadding, int wpadding,
                            int hstride, int wstride,
                            int hdilation, int wdilation)
{{
    int ni, ci, khi, kwi, hoi, woi;
    int idx, hi, wi;

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
        ni = GET_N(idx, n, c, ho, wo);
        hoi = GET_H(idx, n, c, ho, wo);
        woi = GET_W(idx, n, c, ho, wo);
        ci = GET_CI(idx, n, c, ho, wo);
        
        for(khi = 0; khi < kh; khi++)
        {{
            hi = wstride * hoi + hdilation * khi - hpadding;
            for(kwi = 0; (kwi < kw) && IS_BETWEEN(0, hi, h); kwi++)
            {{
                wi = wstride * woi + wdilation * kwi - wpadding;
                if(IS_BETWEEN(0, wi, w))
                    *(y + idx) += (*(x + SHIFT(ni, ci, hi, wi, n, c, h, w))) * (*(weights + SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw)));
            }}
        }}
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler) 
    # ---


    def _bwd_nchw_kernel(self, macros: str) -> np.RawKernel:
        func_name = "depthwise_conv_bwd_nchw"
        return self._bwd_kernel(func_name, macros)
    # ---

    def _bwd_nhwc_kernel(self, macros: str) -> np.RawKernel:
        func_name = "depthwise_conv_bwd_nhwc"
        return self._bwd_kernel(func_name, macros)
    # ---

    def _bwd_kernel(self, func_name: str, macros: str) -> np.RawKernel:
        
        # NOTE: N = n * ci * hi * wo; dx's format: NCHW
        code = \
            r"""
extern "C"
{MACROS}
#define SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw) ((ci * kh + khi) * kw + kwi)
#define GET_C_WEIGHTS(idx, c, kh, kw) (idx / (kw * kh))
#define GET_H_WEIGHTS(idx, c, kh, kw) (idx / kw) % kh
#define GET_W_WEIGHTS(idx, c, kh, kw) (idx % kw)
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void {FUNC_NAME}({T}* dx, {T}* dy, {T}* x,
                            {T}* dw, {T}* weights,
                            int n, int c, int h, int w,
                            int ho, int wo, int kh, int kw,
                            int vpadding, int wpadding,
                            int hstride, int wstride,
                            int hdilation, int wdilation)
{{
    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    const int N = n * h * w * c;
    const int N_W = c * kh * kw;

    int idx, ni, ci, hoi, woi, khi, kwi, hi, wi, _hoi, _woi;
    int n_samples, n_offset, end_offset;
    int samples_worker, samples_overworker, overworkers;

    // Input gradient (dx)

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
            _hoi = (hi + vpadding - hdilation * khi);
            hoi = _hoi / wstride;
            _hoi = _hoi % wstride;

            for(kwi = 0; (kwi < kw) && ((_hoi == 0) && IS_BETWEEN(0, hoi, ho)); kwi++)
            {{
                _woi = (wi + wpadding - wdilation * kwi);
                woi = _woi / wstride;
                _woi = _woi % wstride;

                if((_woi == 0) && IS_BETWEEN(0, woi, wo))
                    *(dx + idx) += (*(weights + SHIFT_WEIGHTS(ci, khi, kwi, n, c, kh, kw))) * (*(dy + SHIFT(ni, ci, hi, wi, n, c, h, w)));
            }}
        }}
    }}

    // Weights gradient (dw)

    overworkers = N_W % num_workers;
    samples_worker = N_W / num_workers;
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
        ci = GET_C_WEIGHTS(idx, c, kh, kw);
        khi = GET_H_WEIGHTS(idx, c, kh, kw);
        kwi = GET_W_WEIGHTS(idx, c, kh, kw);

        for(ni = 0; ni < n; ni++)
        {{
            for(hoi = 0; hoi < ho; hoi++)
            {{
                hi = hstride * hoi + hdilation * khi - hpadding;
                for(woi = 0; (woi < wo) && (IS_BETWEEN(0, hi, h)); woi++)
                {{
                    
                    wi = wstride * woi + wdilation * kwi - wpadding;
                    if(IS_BETWEEN(0, wi, w))
                        *(dw + idx) += (*(weights + SHIFT_WEIGHTS(ci, khi, kwi, c, kh, kw))) * (*(x + SHIFT(ni, ci, hi, wi, n, c, h, w)));
                }}
            }}
        }}
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype], MACROS=macros)
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # ---
