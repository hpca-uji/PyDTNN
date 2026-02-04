import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

__all__ = (
    "im2col_1ch_nchw_cython",
    "col2im_1ch_nchw_cython"
)

# =================== #
# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# =================== #

# =================== #

# --- im2col --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2col_1ch_nchw_cython(npDT[:,:,:,::1] x,
                           npDT[:,::1] cols, 
                           int kh, int kw, int ho, int wo,
                           int vpadding, int hpadding,
                           int vstride, int hstride, 
                           int vdilation, int hdilation) -> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * ho * wo + cc * ho * wo + xx * wo + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]

# ================== #


# ================== #

# --- col2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def col2im_1ch_nchw_cython(npDT[:,::1] cols,
                           npDT[:,:,:,::1] x,
                           int n, int h, int w, int c,
                           int kh, int kw, int ho, int wo,
                           int vpadding, int hpadding,
                           int vstride, int hstride,
                           int vdilation, int hdilation) -> None:

    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * ho * wo + cc * ho * wo + xx * wo + yy
                                    x[nn, cc, x_x, x_y] += cols[row, col]

# ================== #
