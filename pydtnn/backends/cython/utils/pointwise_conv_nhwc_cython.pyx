cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "fwd_pointwise_conv_cython_nhwc",
    "bwd_pointwise_conv_cython_nhwc"
)


def fwd_pointwise_conv_cython_nhwc(npDT[:,:,:,::1] x,
                                   npDT[:,::1] k,
                                   npDT[:,:,:,::1] out,
                                   int hpadding, int wpadding,
                                   int hstride, int wstride,
                                   int hdilation, int wdilation)-> None:

    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int ho = out.shape[1]
    cdef int wo = out.shape[2]
    cdef int co = out.shape[3]

    cdef int nn, cco, cc, ii, jj, x_x, x_y

    for nn in prange(n, nogil=True):
        for ii in range(ho):
            x_x = hstride * ii + (hdilation - 1) - hpadding
            if 0 <= x_x < h:
                for jj in range(wo):
                    x_y = wstride * jj + (wdilation - 1) - wpadding
                    if 0 <= x_y < w:
                        for cco in range(co):
                            for cc in range(c):
                                out[nn, ii, jj, cco] += (k[cc, cco] * x[nn, x_x, x_y, cc])

def bwd_pointwise_conv_cython_nhwc(npDT[:,:,:,::1] dy,
                                   npDT[:,:,:,::1] x,
                                   npDT[:,::1] k,
                                   npDT[:,:,:,::1] dx,
                                   npDT[:,::1] dw,
                                   int hpadding, int wpadding,
                                   int hstride, int wstride,
                                   int hdilation, int wdilation)-> None:

    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int ho = dy.shape[1]
    cdef int wo = dy.shape[2]
    cdef int co = dy.shape[3]

    cdef int nn, cco, cc, yy, xx, x_x, x_y
    cdef npDT val_dy
    
    for nn in prange(n, nogil=True):
        for xx in range(ho):
            x_x = hstride * xx + (hdilation - 1) - hpadding
            if 0 <= x_x < h:
                for yy in range(wo):
                    x_y = wstride * yy + (wdilation - 1) - wpadding
                    if 0 <= x_y < w:
                        for cco in range(co):
                            val_dy = dy[nn, xx, yy, cco]
                            for cc in range(c):
                                dw[cc, cco] += (x[nn, x_x, x_y, cc] * val_dy)
                                dx[nn, x_x, x_y, cc] += (k[cc, cco] * val_dy)
