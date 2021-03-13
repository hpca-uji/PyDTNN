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
#    https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2col_cython.pyx

def im2col_cython(x, int kh, int kw, int vpadding, int hpadding, int vstride, int hstride):
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray x_padded = np.pad(x,
                                      ((0, 0), (0, 0), (vpadding, vpadding), (hpadding, hpadding)),
                                      mode='constant').astype(x.dtype)

    cdef np.ndarray cols = np.zeros((c * kh * kw, n * hh * ww), dtype=x.dtype)

    if x.dtype == np.int8:
        im2col_cython_inner_int8(cols, x_padded, n, c, h, w, hh, ww, kh, kw, vstride, hstride)
    elif x.dtype == np.float32:
        im2col_cython_inner_float32(cols, x_padded, n, c, h, w, hh, ww, kh, kw, vstride, hstride)
    elif x.dtype == np.float64:
        im2col_cython_inner_float64(cols, x_padded, n, c, h, w, hh, ww, kh, kw, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by im2col_cython!".format(str(cols.dtype)))

    return cols

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_3x3_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                                         np.ndarray[np.float32_t, ndim=4] x_padded,
                                         int n, int c, int h, int w, int hh, int ww,
                                         int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col, col1
    for cc in prange(c, nogil=True, schedule='static'):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        for yy in range(ww):
                            col = nn * hh * ww + xx * ww + yy
                            cols[row, col] = x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                                  np.ndarray[np.int8_t, ndim=4] x_padded,
                                  int n, int c, int h, int w, int hh, int ww,
                                  int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        for yy in range(ww):
                            col = nn * hh * ww + xx * ww + yy
                            cols[row, col] = x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                                     np.ndarray[np.float32_t, ndim=4] x_padded,
                                     int n, int c, int h, int w, int hh, int ww,
                                     int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        for yy in range(ww):
                            col = nn * hh * ww + xx * ww + yy
                            cols[row, col] = x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                                     np.ndarray[np.float64_t, ndim=4] x_padded,
                                     int n, int c, int h, int w, int hh, int ww,
                                     int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        for yy in range(ww):
                            col = nn * hh * ww + xx * ww + yy
                            cols[row, col] = x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj]

def col2im_cython(cols,
                  int n, int c, int h, int w,
                  int kh, int kw,
                  int vpadding, int hpadding, int vstride, int hstride):
    cdef int hh = (h + 2 * vpadding - kh) // vstride + 1
    cdef int ww = (w + 2 * hpadding - kw) // hstride + 1

    cdef np.ndarray x_padded = np.zeros((n, c, h + 2 * vpadding, w + 2 * hpadding),
                                        dtype=cols.dtype)
    if cols.dtype == np.int8:
        col2im_cython_inner_int8(cols, x_padded, n, c, h, w, hh, ww, kh, kw, vstride, hstride)
    elif cols.dtype == np.float32:
        col2im_cython_inner_float32(cols, x_padded, n, c, h, w, hh, ww, kh, kw, vstride, hstride)
    elif cols.dtype == np.float64:
        col2im_cython_inner_float64(cols, x_padded, n, c, h, w, hh, ww, kh, kw, vstride, hstride)
    else:
        raise TypeError("Type '{}' is not supported by col2im_cython!".format(str(cols.dtype)))

    if vpadding > 0 or hpadding > 0:
        # @warning: padding:-padding will not work if padding is 0
        return x_padded[:, :, vpadding:vpadding + h, hpadding:hpadding + w]
    return x_padded

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                                  np.ndarray[np.int8_t, ndim=4] x_padded,
                                  int n, int c, int h, int w, int hh, int ww,
                                  int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        # Throw away 1)
                        for yy in range(ww):
                            # Throw away 2)
                            col = nn * hh * ww + xx * ww + yy
                            x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj] += cols[row, col]

#                                   x_x                           x_y
#                           x[n, c, vstride * xx + ii - vpadding, hstride * yy + jj - hpadding] += cols[]
# Throw away 1)
# x_x = vstride * xx + ii - vpadding
# if x_x < 0 or x_x >= H:
#   continue
#
# Throw away 2)
# x_y = hstride * yy + jj - hpadding
# if x_y < 0 or x_y >= W:
#  continue

# Alternative to throw away 1)
# Range for xx: from:  / a >=0
#                      \ vstride * xx + ii - vpadding >= 0
#                         -> a >= (vpadding - ii) // vstride
#                      -> xx = max(0, (vpadding - ii) // vstride))
#
#               to:    / xx < HH
#                      \ vstride * xx + ii - vpadding < H
#                         -> xx < H + (vpadding - ii) // vstride
#                      -> xx = min(HH, H + (vpadding - ii) // vstride))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                                     np.ndarray[np.float32_t, ndim=4] x_padded,
                                     int n, int c, int h, int w, int hh, int ww,
                                     int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        for yy in range(ww):
                            col = nn * hh * ww + xx * ww + yy
                            x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj] += cols[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                                     np.ndarray[np.float64_t, ndim=4] x_padded,
                                     int n, int c, int h, int w, int hh, int ww,
                                     int kh, int kw, int vstride, int hstride) except? -1:
    cdef int cc, ii, jj, row, yy, xx, nn, col

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        for yy in range(ww):
                            col = nn * hh * ww + xx * ww + yy
                            x_padded[nn, cc, vstride * xx + ii, hstride * yy + jj] += cols[row, col]
