#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def max_pool_2d_fwd_nhwc_cython(x, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride):
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray y = np.zeros((n, hh, ww, c), dtype=x.dtype)
    cdef np.ndarray idx_max = np.zeros((n, hh, ww, c), dtype=np.int32)

    if x.dtype == np.int8:
        max_pool_2d_fwd_nhwc_cython_inner_int8(y, x, idx_max, n, h, w, c,
                                               hh, ww, kh, kw, vpadding, hpadding, vstride, hstride)
    elif x.dtype == np.float32:
        max_pool_2d_fwd_nhwc_cython_inner_float32(y, x, idx_max, n, h, w, c,
                                               hh, ww, kh, kw, vpadding, hpadding, vstride, hstride)
    elif x.dtype == np.float64:
        max_pool_2d_fwd_nhwc_cython_inner_float64(y, x, idx_max, n, h, w, c,
                                               hh, ww, kh, kw, vpadding, hpadding, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by max_pool_2d_fwd_nhwc_cython!".format(str(y.dtype)))

    return y, idx_max

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_2d_fwd_nhwc_cython_inner_int8(np.ndarray[np.int8_t, ndim=4] y,
                                                np.ndarray[np.int8_t, ndim=4] x,
                                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                                int n, int h, int w, int c, int hh, int ww,
                                                int kh, int kw, int vpadding, int hpadding,
                                                int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef np.int8_t maxval, minval, val
    minval = np.finfo(np.int8).min

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    maxval, idx_maxval = minval, 0
                    for ii in range(kh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    val = x[nn, x_x, x_y, cc]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * kw + jj
                    y[nn, xx, yy, cc], idx_max[nn, xx, yy, cc] = maxval, idx_maxval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_2d_fwd_nhwc_cython_inner_float32(np.ndarray[np.float32_t, ndim=4] y,
                                                np.ndarray[np.float32_t, ndim=4] x,
                                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                                int n, int h, int w, int c, int hh, int ww,
                                                int kh, int kw, int vpadding, int hpadding,
                                                int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef np.float32_t maxval, minval, val
    minval = np.finfo(np.float32).min

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    maxval, idx_maxval = minval, 0
                    for ii in range(kh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    val = x[nn, x_x, x_y, cc]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * kw + jj
                    y[nn, xx, yy, cc], idx_max[nn, xx, yy, cc] = maxval, idx_maxval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_2d_fwd_nhwc_cython_inner_float64(np.ndarray[np.float64_t, ndim=4] y,
                                                np.ndarray[np.float64_t, ndim=4] x,
                                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                                int n, int h, int w, int c, int hh, int ww,
                                                int kh, int kw, int vpadding, int hpadding,
                                                int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef np.float64_t maxval, minval, val
    minval = np.finfo(np.float64).min

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    maxval, idx_maxval = minval, 0
                    for ii in range(kh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    val = x[nn, x_x, x_y, cc]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * kw + jj
                    y[nn, xx, yy, cc], idx_max[nn, xx, yy, cc] = maxval, idx_maxval

def max_pool_2d_bwd_nhwc_cython(y, idx_max,
                                int n, int h, int w, int c,
                                int kh, int kw,
                                int vpadding, int hpadding, int vstride, int hstride):
    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray x = np.zeros((n, h, w, c), dtype=y.dtype)

    if y.dtype == np.int8:
        max_pool_2d_bwd_nhwc_cython_inner_int8(y, x, idx_max, n, h, w, c,
                                               hh, ww, kh, kw, vpadding, hpadding, vstride, hstride)
    elif y.dtype == np.float32:
        max_pool_2d_bwd_nhwc_cython_inner_float32(y, x, idx_max, n, h, w, c,
                                               hh, ww, kh, kw, vpadding, hpadding, vstride, hstride)
    elif y.dtype == np.float64:
        max_pool_2d_bwd_nhwc_cython_inner_float64(y, x, idx_max, n, h, w, c,
                                               hh, ww, kh, kw, vpadding, hpadding, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by max_pool_2d_bwd_nhwc_cython!".format(str(y.dtype)))

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_2d_bwd_nhwc_cython_inner_int8(np.ndarray[np.int8_t, ndim=4] y,
                                                np.ndarray[np.int8_t, ndim=4] x,
                                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                                int n, int h, int w, int c, int hh, int ww,
                                                int kh, int kw, int vpadding, int hpadding,
                                                int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    idx_maxval = idx_max[nn, xx, yy, cc]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = vstride * xx + ii - vpadding
                    x_y = hstride * yy + jj - hpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        x[nn, x_x, x_y, cc] += y[nn, xx, yy, cc]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_2d_bwd_nhwc_cython_inner_float32(np.ndarray[np.float32_t, ndim=4] y,
                                                np.ndarray[np.float32_t, ndim=4] x,
                                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                                int n, int h, int w, int c, int hh, int ww,
                                                int kh, int kw, int vpadding, int hpadding,
                                                int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    idx_maxval = idx_max[nn, xx, yy, cc]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = vstride * xx + ii - vpadding
                    x_y = hstride * yy + jj - hpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        x[nn, x_x, x_y, cc] += y[nn, xx, yy, cc]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max_pool_2d_bwd_nhwc_cython_inner_float64(np.ndarray[np.float64_t, ndim=4] y,
                                                np.ndarray[np.float64_t, ndim=4] x,
                                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                                int n, int h, int w, int c, int hh, int ww,
                                                int kh, int kw, int vpadding, int hpadding,
                                                int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    idx_maxval = idx_max[nn, xx, yy, cc]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = vstride * xx + ii - vpadding
                    x_y = hstride * yy + jj - hpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        x[nn, x_x, x_y, cc] += y[nn, xx, yy, cc]
