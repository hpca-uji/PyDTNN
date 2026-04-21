cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "im2col_nchw_cython",
    "col2im_nchw_cython",
    "alt_col2im_nchw_cython",

    "im2col_nchw_3x3_cython_inner",
)


# --- im2col --- #

# NOTE:
# This code has been inspired from cthorey, see:
#    https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2col_cython.pyx
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2col_nchw_cython(npDT[:,:,:,::1] x,
                       npDT[:,::1] cols,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation) -> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y
    
    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = (cc * kh + ii) * kw + jj
                for nn in range(n):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        for yy in range(wo):
                            x_y = hstride * yy + hdilation * jj - hpadding
                            col = (nn * ho + xx) * wo + yy
                            if 0 <= x_x < h and 0 <= x_y < w:
                                cols[row, col] = x[nn, cc, x_x, x_y]
                            else:
                                cols[row, col] = <npDT> 0.0
# --- im2col_nchw_cython --- #

# ================== #

# ================== #

# --- col2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def col2im_nchw_cython(npDT[:,::1] cols,
                       npDT[:,:,:,::1] dx,
                       int n, int c, int h, int w,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation) -> None:

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = (cc * kh + ii) * kw + jj
                for nn in range(n):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = (nn * ho + xx) * wo + yy
                                    dx[nn, cc, x_x, x_y] += cols[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def alt_col2im_nchw_cython(npDT[:,::1] cols,
                           npDT[:,:,:,::1] dx,
                           int n, int c, int h, int w,
                           int kh, int kw, int ho, int wo,
                           int vpadding, int hpadding,
                           int vstride, int hstride,
                           int vdilation, int hdilation) -> None:
    # NOTE: A different way to do col2im
    cdef int cc, ii, jj, row, nn, col, x_x, x_y, x_o, y_o, xx, yy

    for nn in prange(n, nogil=True):
        for cc in range(c):
            for x_x in range(h):
                for x_y in range(w):
                    for ii in range(kh):
                        for jj in range(kw):
                            # x_x = vstride * xx + vdilation * ii - vpadding
                            x_o = (x_x + vpadding - vdilation * ii)
                            xx = x_o // vstride
                            x_o = x_o % vstride

                            # x_y = hstride * yy + hdilation * jj - hpadding
                            y_o = (x_y + hpadding - hdilation * jj)
                            yy = y_o // hstride
                            y_o = y_o % hstride
                            
                            if (x_o == 0) and (y_o == 0) and ((0 <= xx < ho) and (0 <= yy < wo)):
                                row = ((cc * kh) + ii) * kw + jj
                                col = ((nn * ho) + xx) * wo + yy
                                dx[nn, cc, x_x, x_y] += cols[row, col]

# ================================== #

# ================================== #


# ---- im2col_nchw_3x3_cython_inner ---- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef im2col_nchw_3x3_cython_inner(npDT[:,::1] cols,
                                  npDT[:,:,:,::1] x,
                                  int n, int c, int h, int w, int ho, int wo,
                                  int kh, int kw, int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation):
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True, schedule='static'):
        for ii in range(kh):
            for jj in range(kw):
                row = (cc * kh + ii) * kw + jj
                for nn in range(n):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        for yy in range(wo):
                            x_y = hstride * yy + hdilation * jj - hpadding
                            col = (nn * ho + xx) * wo + yy
                            if (0 <= x_y < w) and (0 <= x_x < h):
                                cols[row, col] = x[nn, cc, x_x, x_y]
                            else:
                                cols[row, col] = <npDT> 0.0
