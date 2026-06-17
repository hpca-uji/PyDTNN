cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "fwd_pointwise_conv_cython_nchw",
    "bwd_pointwise_conv_cython_nchw"
)


def fwd_pointwise_conv_cython_nchw(npDT[:,:,:,::1] x,
                                   npDT[:,::1] k,
                                   npDT[:,:,:,::1] out,
                                   int hpadding, int wpadding,
                                   int hstride, int wstride)-> None:

    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int co = out.shape[1]
    cdef int ho = out.shape[2]
    cdef int wo = out.shape[3]


    cdef int nn, cco, cc, ii, jj
    cdef int x_x, x_y

    for nn in prange(n, nogil=True):
        for cco in range(co):
            for ii in range(ho):
                x_x = hstride * ii - hpadding
                if 0 <= x_x < h:
                    for jj in range(wo):
                        x_y = wstride * jj - wpadding
                        if 0 <= x_y < w:
                            for cc in range(c):
                                out[nn, cco, ii, jj] += (k[cco, cc] * x[nn, cc, x_x, x_y])

def bwd_pointwise_conv_cython_nchw(npDT[:,:,:,::1] dy,
                                   npDT[:,:,:,::1] x,
                                   npDT[:,::1] k,
                                   npDT[:,:,:,::1] dx,
                                   npDT[:,::1] dw,
                                   int hpadding, int wpadding,
                                   int hstride, int wstride)-> None:

    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int co = dy.shape[1]
    cdef int ho = dy.shape[2]
    cdef int wo = dy.shape[3]

    cdef int nn, cco, cc, yy, xx, x_x, x_y
    cdef npDT val_dy

    for nn in prange(n, nogil=True):
        for cco in range(co):
            for xx in range(ho):
                x_x = hstride * xx - hpadding
                if 0 <= x_x < h:
                    for yy in range(wo):
                        x_y = wstride * yy - wpadding
                        val_dy = dy[nn, cco, xx, yy]
                        if 0 <= x_y < w:
                            for cc in range(c):
                                dw[cco, cc] += (x[nn, cc, x_x, x_y] * val_dy)
                                dx[nn, cc, x_x, x_y] += (k[cco, cc] * val_dy)
