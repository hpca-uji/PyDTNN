import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

__all__ = (
    "im2row_nhwc_cython",
    "row2im_nhwc_cython",
)

# Declare fused type npDT (to be used with template functions)
# =================== #
# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #
# =================== #

# --- im2row --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2row_nhwc_cython(npDT[:,:,:,::1] x,
                       npDT[:,::1] rows,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride, int vdilation, int hdilation) -> None:
    # Initialize variables
    cdef:
        int n = x.shape[0]
        int h = x.shape[1]
        int w = x.shape[2]
        int c = x.shape[3]
    
    #rows = np.zeros((n * ho * wo, c * kh * kw), dtype=x.dtype)

    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y
    if n >= ho:
        for nn in prange(n, nogil=True):
            for xx in range(ho):
                for yy in range(wo):
                    row = nn * ho * wo + xx * wo + yy
                    for cc in range(c):
                        for ii in range(kh):
                            x_x = vstride * xx + vdilation * ii - vpadding
                            if 0 <= x_x < h:
                                for jj in range(kw):
                                    x_y = hstride * yy + hdilation * jj - hpadding
                                    if 0 <= x_y < w:
                                        col = cc * kh * kw + ii * kw + jj
                                        rows[row, col] = x[nn, x_x, x_y, cc]
    else:
        for xx in prange(ho, nogil=True):
            for nn in prange(n):
                for yy in range(wo):
                    row = nn * ho * wo + xx * wo + yy
                    for cc in range(c):
                        for ii in range(kh):
                            x_x = vstride * xx + vdilation * ii - vpadding
                            if 0 <= x_x < h:
                                for jj in range(kw):
                                    x_y = hstride * yy + hdilation * jj - hpadding
                                    if 0 <= x_y < w:
                                        col = cc * kh * kw + ii * kw + jj
                                        rows[row, col] = x[nn, x_x, x_y, cc]
# --- END im2row_nhwc_cython --- #
                                
# --- END im2row --- #


# ========================================== #


# ========================================== #


# --- row2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def row2im_nhwc_cython(npDT[:,::1] rows,
                       npDT[:,:,:,::1] x,
                       int n, int h, int w, int c,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation) -> None: 
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    if n >= ho:
        for nn in prange(n, nogil=True):
            for xx in range(ho):
                for yy in range(wo):
                    row = nn * ho * wo + xx * wo + yy
                    for cc in range(c):
                        for ii in range(kh):
                            x_x = vstride * xx + vdilation * ii - vpadding
                            if 0 <= x_x < h:
                                for jj in range(kw):
                                    x_y = hstride * yy + hdilation * jj - hpadding
                                    if 0 <= x_y < w:
                                        col = cc * kh * kw + ii * kw + jj
                                        x[nn, x_x, x_y, cc] += rows[row, col]
    else:
        for xx in prange(ho, nogil=True):
            for nn in range(n):
                for yy in range(wo):
                    row = nn * ho * wo + xx * wo + yy
                    for cc in range(c):
                        for ii in range(kh):
                            x_x = vstride * xx + vdilation * ii - vpadding
                            if 0 <= x_x < h:
                                for jj in range(kw):
                                    x_y = hstride * yy + hdilation * jj - hpadding
                                    if 0 <= x_y < w:
                                        col = cc * kh * kw + ii * kw + jj
                                        x[nn, x_x, x_y, cc] += rows[row, col]
# --- END row2im_nhwc_cython --- #

# --- END row2im --- #
# ========================================== #
