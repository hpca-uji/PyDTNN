import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange

from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "max_pool_2d_fwd_nchw_cython",
    "max_pool_2d_bwd_nchw_cython"
)


# --- Forward --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def max_pool_2d_fwd_nchw_cython(npDT[:,:,:,::1] x,
                                npDT[:,:,:,::1] y,
                                np.int32_t[:,:,:,::1] idx_max,
                                int kh, int kw, int ho, int wo,
                                int hpadding, int wpadding,
                                int hstride, int wstride,
                                int hdilation, int wdilation,
                                npDT minval) -> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef npDT maxval, val

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for xx in range(ho):
                for yy in range(wo):
                    maxval, idx_maxval = minval, 0
                    for ii in range(kh):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    val = x[nn, cc, x_x, x_y]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * kw + jj
                    y[nn, cc, xx, yy], idx_max[nn, cc, xx, yy] = maxval, idx_maxval


# =================== #

# =================== #


# --- Backward --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def max_pool_2d_bwd_nchw_cython(npDT[:,:,:,::1] dy,
                                np.int32_t[:,:,:,::1] idx_max,
                                npDT[:,:,:,::1] dx,
                                int n, int h, int w, int c,
                                int kh, int kw, int ho, int wo,
                                int hpadding, int wpadding,
                                int hstride, int wstride,
                                int hdilation, int wdilation) -> None:

    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for xx in range(ho):
                for yy in range(wo):
                    idx_maxval = idx_max[nn, cc, xx, yy]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = hstride * xx + hdilation * ii - hpadding
                    x_y = wstride * yy + wdilation * jj - wpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        dx[nn, cc, x_x, x_y] += dy[nn, cc, xx, yy]


def max_pool_2d_bwd_alt(npDT[:,:,:,::1] dy,
                        np.int32_t[:,:,:,::1] idx_max,
                        npDT[:,:,:,::1] dx,
                        int n, int h, int w, int c,
                        int kh, int kw, int ho, int wo,
                        int hpadding, int wpadding,
                        int hstride, int wstride,
                        int hdilation, int wdilation) -> None:

    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval
    cdef int _xx, _yy, hi, wi, khi, kwi

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(h):
                for wi in range(w):
                    for khi in range(kh):
                        for kwi in range(kw):
                            _xx = hi + hpadding - hdilation * khi
                            xx = _xx // hstride
                            _xx = _xx  % hstride

                            _yy = wi + wpadding - wdilation * kwi
                            yy = _yy // wstride
                            _yy = _yy % wstride

                            if _xx == 0 and _yy == 0 and 0 <= xx < ho and 0 <= yy < wo:
                                idx_maxval = idx_max[nn, cc, xx, yy]
                                ii, jj = idx_maxval // kh, idx_maxval % kw
                                if (ii == khi and jj == kwi):
                                    dx[nn, cc, hi, wi] += dy[nn, cc, xx, yy]