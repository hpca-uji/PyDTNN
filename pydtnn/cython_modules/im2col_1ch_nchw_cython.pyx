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

def im2col_1ch_nchw_cython(x, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride):
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray cols = np.zeros((kh * kw, n * c * hh * ww), dtype=x.dtype)

    if x.dtype == np.int8:
        im2col_1ch_nchw_cython_inner_int8(cols, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding, vstride, hstride)
    elif x.dtype == np.float32:
        im2col_1ch_nchw_cython_inner_float32(cols, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding, vstride, hstride)
    elif x.dtype == np.float64:
        im2col_1ch_nchw_cython_inner_float64(cols, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by im2col_1ch_nchw_cython!".format(str(cols.dtype)))

    return cols

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_1ch_nchw_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                                           np.ndarray[np.int8_t, ndim=4] x,
                                           int n, int h, int w, int c, int hh, int ww,
                                           int kh, int kw, int vpadding, int hpadding,
                                           int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(hh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * hh * ww + cc * hh * ww + xx * ww + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_1ch_nchw_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                                           np.ndarray[np.float32_t, ndim=4] x,
                                           int n, int h, int w, int c, int hh, int ww,
                                           int kh, int kw, int vpadding, int hpadding,
                                           int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(hh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * hh * ww + cc * hh * ww + xx * ww + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_1ch_nchw_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                                           np.ndarray[np.float64_t, ndim=4] x,
                                           int n, int h, int w, int c, int hh, int ww,
                                           int kh, int kw, int vpadding, int hpadding,
                                           int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(hh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * hh * ww + cc * hh * ww + xx * ww + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]

def col2im_1ch_nchw_cython(cols,
                           int n, int h, int w, int c,
                           int kh, int kw,
                           int vpadding, int hpadding, int vstride, int hstride):
    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray x = np.zeros((n, c, h, w), dtype=cols.dtype)

    if cols.dtype == np.int8:
        col2im_1ch_nchw_cython_inner_int8(cols, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding, vstride, hstride)
    elif cols.dtype == np.float32:
        col2im_1ch_nchw_cython_inner_float32(cols, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding, vstride, hstride)
    elif cols.dtype == np.float64:
        col2im_1ch_nchw_cython_inner_float64(cols, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by col2im_1ch_nchw_cython!".format(str(cols.dtype)))

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_1ch_nchw_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                                       np.ndarray[np.int8_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(hh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * hh * ww + cc * hh * ww + xx * ww + yy
                                    x[nn, cc, x_x, x_y] += cols[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_1ch_nchw_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                                       np.ndarray[np.float32_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(hh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * hh * ww + cc * hh * ww + xx * ww + yy
                                    x[nn, cc, x_x, x_y] += cols[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_1ch_nchw_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                                       np.ndarray[np.float64_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for ii in prange(kh, nogil=True):
        for jj in range(kw):
            row = ii * kw + jj
            for nn in range(n):
                for cc in range(c):
                    for xx in range(hh):
                        x_x = vstride * xx + ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * c * hh * ww + cc * hh * ww + xx * ww + yy
                                    x[nn, cc, x_x, x_y] += cols[row, col]
