import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "im2row_nhwc_cython",
    "row2im_nhwc_cython",
    "alt_row2im_nhwc_cython"
)


# --- im2row --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2row_nhwc_cython(npDT[:,:,:,::1] x,
                       npDT[:,::1] rows,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride, 
                       int vdilation, int hdilation) -> None:
    # Initialize variables
    cdef:
        int n = x.shape[0]
        int h = x.shape[1]
        int w = x.shape[2]
        int c = x.shape[3]
    
    #rows = np.zeros((n * ho * wo, c * kh * kw), dtype=x.dtype)

    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y
    # if n >= ho:
    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                row = (nn * ho + xx) * wo + yy
                for ii in range(kh):
                    x_x = vstride * xx + vdilation * ii - vpadding
                    for jj in range(kw):
                        x_y = hstride * yy + hdilation * jj - hpadding
                        for cc in range(c):
                            col = (cc * kh + ii) * kw + jj
                            if (0 <= x_x < h) and (0 <= x_y < w):
                                rows[row, col] = x[nn, x_x, x_y, cc]
                            else:
                                rows[row, col] = <npDT> 0.0

    # FIXME: Optimization broken (maybe)
    # else:
    #     for xx in prange(ho, nogil=True):
    #         for nn in prange(n):
    #             for yy in range(wo):
    #                 row = nn * ho * wo + xx * wo + yy
    #                 for cc in range(c):
    #                     for ii in range(kh):
    #                         x_x = vstride * xx + vdilation * ii - vpadding
    #                         if 0 <= x_x < h:
    #                             for jj in range(kw):
    #                                 x_y = hstride * yy + hdilation * jj - hpadding
    #                                 if 0 <= x_y < w:
    #                                     col = cc * kh * kw + ii * kw + jj
    #                                     rows[row, col] = x[nn, x_x, x_y, cc]
                                


# ========================================== #


# ========================================== #


# --- row2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def row2im_nhwc_cython(npDT[:,::1] rows,
                       npDT[:,:,:,::1] dx,
                       int n, int h, int w, int c,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation) -> None: 
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    # if n >= ho:
    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                row = (nn * ho + xx) * wo + yy
                for cc in range(c):
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = (cc * kh + ii) * kw + jj
                                    dx[nn, x_x, x_y, cc] += rows[row, col]
    # FIXME: Optimization broken
    # else:
    #     for xx in prange(ho, nogil=True):
    #         for nn in range(n):
    #             for yy in range(wo):
    #                 row = nn * ho * wo + xx * wo + yy
    #                 for cc in range(c):
    #                     for ii in range(kh):
    #                         x_x = vstride * xx + vdilation * ii - vpadding
    #                         if 0 <= x_x < h:
    #                             for jj in range(kw):
    #                                 x_y = hstride * yy + hdilation * jj - hpadding
    #                                 if 0 <= x_y < w:
    #                                     col = cc * kh * kw + ii * kw + jj
    #                                     x[nn, x_x, x_y, cc] += rows[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def alt_row2im_nhwc_cython(npDT[:,::1] rows,
                           npDT[:,:,:,::1] dx,
                           int n, int h, int w, int c,
                           int kh, int kw, int ho, int wo,
                           int vpadding, int hpadding,
                           int vstride, int hstride,
                           int vdilation, int hdilation) -> None: 
    cdef int nn, row, cc, ii, jj, col, x_x, x_y, x_o, y_o, xx, yy

    for nn in prange(n, nogil=True):
        for x_x in range(h):
            for x_y in range(w):
                for cc in range(c):
                    for ii in range(kh):
                        for jj in range(kw):
                            # x_x = vstride * xx + vdilation * ii - vpadding
                            x_o = x_x + vpadding - vdilation * ii
                            xx = x_o // vstride
                            x_o = x_o % vstride
                            
                            # x_y = hstride * yy + hdilation * jj - hpadding
                            y_o = x_y + hpadding - hdilation * jj
                            yy = y_o // hstride
                            y_o = y_o % hstride


                            if (x_o == 0) and (y_o == 0) and ((0 <= xx < ho) and (0 <= yy < wo)):
                                row = nn * ho * wo + xx * wo + yy
                                col = cc * kh * kw + ii * kw + jj
                                dx[nn, x_x, x_y, cc] += rows[row, col]
# ========================================== #
