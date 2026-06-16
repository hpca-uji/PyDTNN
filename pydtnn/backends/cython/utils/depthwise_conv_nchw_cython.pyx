cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "depthwise_conv_nchw_cython",
    "depthwise_conv_backward_nchw_cython"
)


# --- FORWARD ---
def depthwise_conv_nchw_cython(npDT[:,:,:,::1] x,
                               npDT[:,:,::1] k,
                               npDT[:,:,:,::1] res,
                               int ho, int wo,
                               int hpadding, int wpadding,
                               int hstride, int wstride,
                               int hdilation, int wdilation)-> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int kh = k.shape[1]
    cdef int kw = k.shape[2]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                for nn in range(n):
                    for xx in range(ho):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                if 0 <= x_y < w:
                                    res[nn, cc, xx, yy] += k[cc, ii, jj] * x[nn, cc, x_x, x_y]

# ----- BACKWARD ----
def depthwise_conv_backward_nchw_cython(npDT[:,:,:,::1] dy,
                                        npDT[:,:,:,::1] x,
                                        npDT[:,:,::1] k,
                                        npDT[:,:,:,::1] dx,
                                        npDT[:,:,::1] dw,
                                        int hpadding, int wpadding,
                                        int hstride, int wstride,
                                        int hdilation, int wdilation)-> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int kh = k.shape[1]
    cdef int kw = k.shape[2]

    cdef int ho = dy.shape[2]
    cdef int wo = dy.shape[3]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y
    cdef npDT val_k, val_dy

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                val_k = k[cc, ii, jj]
                for nn in range(n):
                    for xx in range(ho):
                        x_x = hstride * xx + hdilation * ii - hpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = wstride * yy + wdilation * jj - wpadding
                                val_dy = dy[nn, cc, xx, yy]
                                if 0 <= x_y < w:
                                    dw[cc, ii, jj] += x[nn, cc, x_x, x_y] * val_dy
                                    dx[nn, cc, x_x, x_y] += val_k * val_dy
