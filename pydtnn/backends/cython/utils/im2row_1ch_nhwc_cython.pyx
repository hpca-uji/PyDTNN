cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "im2row_1ch_nhwc_cython",
    "row2im_1ch_nhwc_cython"
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2row_1ch_nhwc_cython(npDT[:,:,:,::1] x, 
                           npDT[:,::1] rows,
                           int kh, int kw, int ho, int wo,
                           int vpadding, int hpadding,
                           int vstride, int hstride, int vdilation, int hdilation) -> None:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    row = nn * ho * wo * c + xx * wo * c + yy * c + cc
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    rows[row, col] = x[nn, x_x, x_y, cc]





@cython.boundscheck(False)
@cython.wraparound(False)
def row2im_1ch_nhwc_cython(npDT[:,::1] rows,
                           npDT[:,:,:,::1] x,
                           int n, int h, int w, int c,
                           int kh, int kw,
                           int ho, int wo,
                           int vpadding, int hpadding,
                           int vstride, int hstride,
                           int vdilation, int hdilation) -> None:

    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    row = nn * ho * wo * c + xx * wo * c + yy * c + cc
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    x[nn, x_x, x_y, cc] += rows[row, col]

