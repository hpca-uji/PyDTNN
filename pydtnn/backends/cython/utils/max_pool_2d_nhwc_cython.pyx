import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "max_pool_2d_fwd_nhwc_cython",
    "max_pool_2d_bwd_nhwc_cython"
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def max_pool_2d_fwd_nhwc_cython(npDT[:,:,:,::1] x,
                                npDT[:,:,:,::1] y,
                                np.int32_t[:,:,:,::1] idx_max,
                                int kh, int kw, int ho, int wo,
                                int hpadding, int wpadding,
                                int hstride, int wstride, 
                                int hdilation, int wdilation,
                                npDT minval) -> None:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef npDT maxval, val

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    maxval, idx_maxval = minval, 0
                    for ii in range(kh):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    val = x[nn, x_x, x_y, cc]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * kw + jj
                    y[nn, xx, yy, cc] = maxval
                    idx_max[nn, xx, yy, cc] = idx_maxval
                    


# =================== #

# =================== #


# --- Backward --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def max_pool_2d_bwd_nhwc_cython(npDT[:,:,:,::1] dy,
                                np.int32_t[:,:,:,::1] idx_max,
                                npDT[:,:,:,::1] dx,
                                int n, int h, int w, int c,
                                int kh, int kw, int ho, int wo,
                                int hpadding, int wpadding,
                                int hstride, int wstride,
                                int hdilation, int wdilation) -> None:

    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    idx_maxval = idx_max[nn, xx, yy, cc]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = hstride * xx + hdilation * ii - hpadding
                    x_y = wstride * yy + wdilation * jj - wpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        dx[nn, x_x, x_y, cc] += dy[nn, xx, yy, cc]
