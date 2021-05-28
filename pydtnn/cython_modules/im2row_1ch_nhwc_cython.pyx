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

# This code has been inspired from cthorey, see:
#    https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2row_1ch_nhwc_cython.pyx

def im2row_1ch_nhwc_cython(x, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride):
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray x_padded = np.pad(x,
                                      ((0, 0), (vpadding, vpadding), (hpadding, hpadding), (0, 0)),
                                      mode='constant').astype(x.dtype)

    cdef np.ndarray rows = np.zeros((n * c * hh * ww, kh * kw), dtype=x.dtype)

    if x.dtype == np.int8:
        im2row_1ch_nhwc_cython_inner_int8(rows, x_padded, n, h, w, c, hh, ww, kh, kw, vstride, hstride)
    elif x.dtype == np.float32:
        im2row_1ch_nhwc_cython_inner_float32(rows, x_padded, n, h, w, c, hh, ww, kh, kw, vstride, hstride)
    elif x.dtype == np.float64:
        im2row_1ch_nhwc_cython_inner_float64(rows, x_padded, n, h, w, c, hh, ww, kh, kw, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by im2row_1ch_nhwc_cython!".format(str(rows.dtype)))

    return rows

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2row_1ch_nhwc_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] rows,
                                           np.ndarray[np.int8_t, ndim=4] x_padded,
                                           int n, int h, int w, int c, int hh, int ww,
                                           int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        for jj in range(kw):
                            col = ii * kw + jj
                            rows[row, col] = x_padded[nn, vstride * xx + ii, hstride * yy + jj, cc]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2row_1ch_nhwc_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] rows,
                                              np.ndarray[np.float32_t, ndim=4] x_padded,
                                              int n, int h, int w, int c, int hh, int ww,
                                              int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        for jj in range(kw):
                            col = ii * kw + jj
                            rows[row, col] = x_padded[nn, vstride * xx + ii, hstride * yy + jj, cc]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2row_1ch_nhwc_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] rows,
                                              np.ndarray[np.float64_t, ndim=4] x_padded,
                                              int n, int h, int w, int c, int hh, int ww,
                                              int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        for jj in range(kw):
                            col = ii * kw + jj
                            rows[row, col] = x_padded[nn, vstride * xx + ii, hstride * yy + jj, cc]

def row2im_1ch_nhwc_cython(rows,
                           int n, int h, int w, int c,
                           int kh, int kw,
                           int vpadding, int hpadding, int vstride, int hstride):
    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray x_padded = np.zeros((n, h + 2 * vpadding, w + 2 * hpadding, c),
                                        dtype=rows.dtype)
    if rows.dtype == np.int8:
        row2im_1ch_nhwc_cython_inner_int8(rows, x_padded, n, h, w, c, hh, ww, kh, kw, vstride, hstride)
    elif rows.dtype == np.float32:
        row2im_1ch_nhwc_cython_inner_float32(rows, x_padded, n, h, w, c, hh, ww, kh, kw, vstride, hstride)
    elif rows.dtype == np.float64:
        row2im_1ch_nhwc_cython_inner_float64(rows, x_padded, n, h, w, c, hh, ww, kh, kw, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by row2im_1ch_nhwc_cython!".format(str(rows.dtype)))

    if vpadding > 0 or hpadding > 0:
        # @warning: padding:-padding will not work if padding is 0
        return x_padded[:, vpadding:vpadding + h, hpadding:hpadding + w, :]
    return x_padded

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_1ch_nhwc_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] rows,
                                           np.ndarray[np.int8_t, ndim=4] x_padded,
                                           int n, int h, int w, int c, int hh, int ww,
                                           int kh, int kw, int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        for jj in range(kw):
                            col = ii * kw + jj
                            x_padded[nn, vstride * xx + ii, hstride * yy + jj, cc] += rows[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_1ch_nhwc_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] rows,
                                              np.ndarray[np.float32_t, ndim=4] x_padded,
                                              int n, int h, int w, int c, int hh, int ww,
                                              int kh, int kw, int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        for jj in range(kw):
                            col = ii * kw + jj
                            x_padded[nn, vstride * xx + ii, hstride * yy + jj, cc] += rows[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_1ch_nhwc_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] rows,
                                              np.ndarray[np.float64_t, ndim=4] x_padded,
                                              int n, int h, int w, int c, int hh, int ww,
                                              int kh, int kw, int vstride, int hstride) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        for jj in range(kw):
                            col = ii * kw + jj
                            x_padded[nn, vstride * xx + ii, hstride * yy + jj, cc] += rows[row, col]
