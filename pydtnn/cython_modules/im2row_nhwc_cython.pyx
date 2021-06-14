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

def im2row_nhwc_cython(x, int kh, int kw, int vpadding, int hpadding,
                       int vstride, int hstride, int vdilation, int hdilation):
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef np.ndarray rows = np.zeros((n * hh * ww, c * kh * kw), dtype=x.dtype)

    if x.dtype == np.int8:
        im2row_nhwc_cython_inner_int8(rows, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                                      vstride, hstride, vdilation, hdilation)
    elif x.dtype == np.float32:
        im2row_nhwc_cython_inner_float32(rows, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                                      vstride, hstride, vdilation, hdilation)
    elif x.dtype == np.float64:
        im2row_nhwc_cython_inner_float64(rows, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                                      vstride, hstride, vdilation, hdilation)
    else:
        raise TypeError("Type '{}' is not supported by im2row_nhwc_cython!".format(str(rows.dtype)))

    return rows

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2row_nhwc_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] rows,
                                       np.ndarray[np.int8_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride,
                                       int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                row = nn * hh * ww + xx * ww + yy
                for cc in range(c):
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
cdef int im2row_nhwc_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] rows,
                                       np.ndarray[np.float32_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride,
                                       int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                row = nn * hh * ww + xx * ww + yy
                for cc in range(c):
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
cdef int im2row_nhwc_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] rows,
                                       np.ndarray[np.float64_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride,
                                       int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                row = nn * hh * ww + xx * ww + yy
                for cc in range(c):
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    rows[row, col] = x[nn, x_x, x_y, cc]

def row2im_nhwc_cython(rows,
                  int n, int h, int w, int c,
                  int kh, int kw,
                  int vpadding, int hpadding,
                  int vstride, int hstride,
                  int vdilation, int hdilation):
    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef np.ndarray x = np.zeros((n, h, w, c), dtype=rows.dtype)

    if rows.dtype == np.int8:
        row2im_nhwc_cython_inner_int8(rows, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                                      vstride, hstride, vdilation, hdilation)
    elif rows.dtype == np.float32:
        row2im_nhwc_cython_inner_float32(rows, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                                      vstride, hstride, vdilation, hdilation)
    elif rows.dtype == np.float64:
        row2im_nhwc_cython_inner_float64(rows, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                                      vstride, hstride, vdilation, hdilation)
    else:
        raise TypeError("Type '{}' is not supported by row2im_nhwc_cython!".format(str(rows.dtype)))

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_nhwc_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] rows,
                                       np.ndarray[np.int8_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride,
                                       int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                row = nn * hh * ww + xx * ww + yy
                for cc in range(c):
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    x[nn, x_x, x_y, cc] += rows[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_nhwc_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] rows,
                                       np.ndarray[np.float32_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride,
                                       int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                row = nn * hh * ww + xx * ww + yy
                for cc in range(c):
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    x[nn, x_x, x_y, cc] += rows[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_nhwc_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] rows,
                                       np.ndarray[np.float64_t, ndim=4] x,
                                       int n, int h, int w, int c, int hh, int ww,
                                       int kh, int kw, int vpadding, int hpadding,
                                       int vstride, int hstride,
                                       int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                row = nn * hh * ww + xx * ww + yy
                for cc in range(c):
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    x[nn, x_x, x_y, cc] += rows[row, col]
