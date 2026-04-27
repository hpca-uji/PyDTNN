cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "average_pool_2d_fwd_nchw_cython",
    "average_pool_2d_bwd_nchw_cython"
)


# --- FORWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_pool_2d_fwd_nchw_cython(npDT[:,:,:,::1] x,
                                    npDT[:,:,:,::1] y,
                                    int kh, int kw, int ho, int wo,
                                    int hpadding, int wpadding,
                                    int hstride, int wstride,
                                    int hdilation, int wdilation) -> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, items
    cdef npDT accum

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for xx in range(ho):
                for yy in range(wo):
                    accum = <npDT> 0.0
                    items = 0
                    # accum, items = 0, (kh * kw)
                    for ii in range(kh):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    accum = accum + x[nn, cc, x_x, x_y]
                                    items = items + 1
                    y[nn, cc, xx, yy] = <npDT> (accum / items)

# =================== #
# =================== #

# --- BACKWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_pool_2d_bwd_nchw_cython(npDT[:,:,:,::1] dy,
                                    npDT[:,:,:,::1] dx,
                                    int n, int h, int w, int c,
                                    int kh, int kw, int ho, int wo,
                                    int hpadding, int wpadding,
                                    int hstride, int wstride,
                                    int hdilation, int wdilation) -> None:
    
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, items
    cdef npDT avgval

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for xx in range(ho):
                for yy in range(wo):
                    items = 0
                    avgval = dy[nn, cc, xx, yy]
                    for ii in range(kh):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    items = items + 1
                    avgval /= items
                    for ii in range(kh):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    dx[nn, cc, x_x, x_y] += avgval


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_pool_2d_bwd_nchw_cython_alt(npDT[:,:,:,::1] dy,
                                        npDT[:,:,:,::1] dx,
                                        int n, int h, int w, int c,
                                        int kh, int kw, int ho, int wo,
                                        int hpadding, int wpadding,
                                        int hstride, int wstride,
                                        int hdilation, int wdilation) -> None:
    
    cdef int nn, xx, yy, cc, ii, jj, _xx, _yy, items, hi, wi, x_x, x_y

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for xx in range(ho):
                for yy in range(wo):
                    items = 0
                    for ii in range(kh):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    items = items + 1
                    dy[nn, cc, xx, yy] = <npDT> (dy[nn, cc, xx, yy] / items)
    
    for nn in prange(n, nogil=True):
        for cc in range(c):
            for hi in range(h):
                for wi in range(w):
                    for ii in range(kh):
                        for jj in range(kw):
                            # hi = hstride * xx + hdilation * ii - hpadding
                            # wi = wstride * yy + wdilation * jj - wpadding
                            _xx = hi + hpadding - hdilation * ii
                            xx = _xx // hstride
                            _xx = _xx % hstride 

                            _yy = wi + wpadding - wdilation * jj
                            yy = _yy // wstride
                            _yy = _yy % wstride

                            if (_xx == 0) and (_yy == 0) and (0 <= xx < ho) and (0 <= yy < wo):
                                dx[nn, cc, hi, wi] = dx[nn, cc, hi, wi] + dy[nn, cc, xx, yy]