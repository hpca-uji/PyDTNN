#
#  This file is part of Python Distributed Training of neural networks (PyDTnn)
#
#  copyright (c) 2021 Universitat Jaume I
#
#  PyDTnn is free software: you can redistribute it and/or modify it under the
#  terms of the GnU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but wIThOUT
#  AnY wARRAnTY; without even the implied warranty of MERchAnTABILITY
#  or FITnESS FOR A PARTIcULAR PURPOSE.  See the GnU General Public
#  License for more details.
#
#  You should have received a copy of the GnU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def depthwise_conv_cython(x, k, int vpadding, int hpadding, 
                          int vstride, int hstride, int vdilation, int hdilation):
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int kh = k.shape[1]
    cdef int kw = k.shape[2]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef np.ndarray res = np.zeros((n, c, hh, ww), dtype=x.dtype)

    if (x.dtype == np.int8):
        depthwise_conv_cython_inner_int8(res, x, k, n, c, h, w,
                                 hh, ww, kh, kw, vpadding, hpadding,
                                 vstride, hstride, vdilation, hdilation)
    elif (x.dtype == np.float32):
        depthwise_conv_cython_inner_float32(res, x, k, n, c, h, w,
                                 hh, ww, kh, kw, vpadding, hpadding,
                                 vstride, hstride, vdilation, hdilation)
    elif (x.dtype == np.float64):
        depthwise_conv_cython_inner_float64(res, x, k, n, c, h, w,
                                 hh, ww, kh, kw, vpadding, hpadding,
                                 vstride, hstride, vdilation, hdilation)
    else:
        raise TypeError("Type '{}' is not supported by depthwise_conv_cython!".format(str(res.dtype)))

    return res

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int depthwise_conv_cython_inner_int8(np.ndarray[np.int8_t, ndim=4] res,
                             np.ndarray[np.int8_t, ndim=4] x,
                             np.ndarray[np.int8_t, ndim=3] k,
                             int n, int c, int h, int w, int hh, int ww,
                             int kh, int kw, int vpadding, int hpadding,
                             int vstride, int hstride,
                             int vdilation, int hdilation) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                for nn in range(n):
                    for xx in range(hh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    res[nn, cc, xx, yy] += k[cc, ii, jj] * x[nn, cc, x_x, x_y]
        
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int depthwise_conv_cython_inner_float32(np.ndarray[np.float32_t, ndim=4] res,
                             np.ndarray[np.float32_t, ndim=4] x,
                             np.ndarray[np.float32_t, ndim=3] k,
                             int n, int c, int h, int w, int hh, int ww,
                             int kh, int kw, int vpadding, int hpadding,
                             int vstride, int hstride,
                             int vdilation, int hdilation) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                for nn in range(n):
                    for xx in range(hh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    res[nn, cc, xx, yy] += k[cc, ii, jj] * x[nn, cc, x_x, x_y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int depthwise_conv_cython_inner_float64(np.ndarray[np.float64_t, ndim=4] res,
                             np.ndarray[np.float64_t, ndim=4] x,
                             np.ndarray[np.float64_t, ndim=3] k,
                             int n, int c, int h, int w, int hh, int ww,
                             int kh, int kw, int vpadding, int hpadding,
                             int vstride, int hstride,
                             int vdilation, int hdilation) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                for nn in range(n):
                    for xx in range(hh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    res[nn, cc, xx, yy] += k[cc, ii, jj] * x[nn, cc, x_x, x_y]
