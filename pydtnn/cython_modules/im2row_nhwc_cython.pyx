#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t


def im2row_nhwc_cython(npDT[:, :, :, :] x, int kh, int kw, int vpadding, int hpadding,
                       int vstride, int hstride, int vdilation, int hdilation):
    # Initialize variables
    cdef:
        int n = x.shape[0]
        int h = x.shape[1]
        int w = x.shape[2]
        int c = x.shape[3]
        int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
    # Initialize dtype (only one of these branches will be compiled for each npDT)
    if npDT is np.int8_t:
        dtype = np.int8
    elif npDT is np.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    # Initialize rows and its view
    cdef:
        np.ndarray rows = np.zeros((n * hh * ww, c * kh * kw), dtype=dtype)
        npDT[:, ::1] rows_view = rows  # 2D contiguous view of rows
    # Call im2row_nhwc_cython_inner
    im2row_nhwc_cython_inner(rows_view, x, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                             vstride, hstride, vdilation, hdilation)
    # Return rows
    return rows


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2row_nhwc_cython_inner(npDT[:, ::1] rows,
                                  npDT[:, :, :, :] x,
                                  int n, int h, int w, int c, int hh, int ww,
                                  int kh, int kw, int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y
    if n >= hh:
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
    else:
        for xx in prange(hh, nogil=True):
            for nn in prange(n):
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


def row2im_nhwc_cython(npDT[:, :] rows,
                       int n, int h, int w, int c,
                       int kh, int kw,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation):
    # Initialize variables
    cdef:
        int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
        int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1
    # Initialize dtype (only one of these branches will be compiled for each npDT)
    if npDT is np.int8_t:
        dtype = np.int8
    elif npDT is np.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    # Initialize x and its view
    cdef:
        np.ndarray x = np.zeros((n, h, w, c), dtype=dtype)
        npDT[:, :, :, ::1] x_view = x  # 4D contiguous view of x
    # Call row2im_nhwc_cython_inner
    row2im_nhwc_cython_inner(rows, x_view, n, h, w, c, hh, ww, kh, kw, vpadding, hpadding,
                             vstride, hstride, vdilation, hdilation)
    # Return x
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int row2im_nhwc_cython_inner(npDT [:, :] rows,
                                  npDT [:, :, :, ::1] x,
                                  int n, int h, int w, int c, int hh, int ww,
                                  int kh, int kw, int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation) except? -1:
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y
    if n >= hh:
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
    else:
        for xx in prange(hh, nogil=True):
            for nn in range(n):
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
