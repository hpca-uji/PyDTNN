#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
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

# =================== #
# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #
# =================== #

# --- im2row --- #

def im2row_1ch_nhwc_cython(x: np.ndarray,
                           int kh, int kw, int vpadding, int hpadding,
                           int vstride, int hstride, int vdilation, int hdilation) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    rows: np.ndarray = np.empty((n * c * hh * ww, kh * kw), dtype=x.dtype)

    try:
        im2row_1ch_nhwc_cython_inner(rows, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding,
                                     vstride, hstride, vdilation, hdilation)
    except TypeError as e:
        raise TypeError(f"Function: \"im2row_1ch_nhwc_cython\". Error: {e}")

    return rows

def im2row_1ch_nhwc_cython_inner(np.ndarray[npDT, ndim=2] rows,
                                 np.ndarray[npDT, ndim=4] x,
                                 int n, int h, int w, int c, int hh, int ww,
                                 int kh, int kw, int vpadding, int hpadding,
                                 int vstride, int hstride,
                                 int vdilation, int hdilation):

    cdef npDT[:,:] rows_view = rows
    cdef npDT[:,:,:,:] x_view = x

    _im2row_1ch_nhwc_cython_inner(rows_view, x_view, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding,
                                  vstride, hstride, vdilation, hdilation)
# --- im2row_1ch_nhwc_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _im2row_1ch_nhwc_cython_inner(npDT[:,:] rows,
                                   npDT[:,:,:,:] x,
                                   int n, int h, int w, int c, int hh, int ww,
                                   int kh, int kw, int vpadding, int hpadding,
                                   int vstride, int hstride,
                                   int vdilation, int hdilation):
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    rows[row, col] = x[nn, x_x, x_y, cc]
# --- _im2row_1ch_nhwc_cython_inner --- #

# --- END im2row --- #


# ================== #

# ================== #


# --- row2im --- #

def row2im_1ch_nhwc_cython(np.ndarray[npDT, ndim=2] rows,
                           int n, int h, int w, int c,
                           int kh, int kw,
                           int vpadding, int hpadding,
                           int vstride, int hstride,
                           int vdilation, int hdilation) -> np.ndarray:
    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef np.ndarray[npDT, ndim=4] x = np.zeros((n, h, w, c), dtype=rows.dtype)

    try:
        row2im_1ch_nhwc_cython_inner(rows, x, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding,
                                     vstride, hstride, vdilation, hdilation)
    except TypeError as e:
        raise TypeError(f"Function: \"row2im_1ch_nhwc_cython\". Error: {e}")

    return x
# --- END row2im_1ch_nhwc_cython --- #

def row2im_1ch_nhwc_cython_inner(np.ndarray[npDT, ndim=2] rows,
                                 np.ndarray[npDT, ndim=4] x,
                                 int n, int h, int w, int c, int hh, int ww,
                                 int kh, int kw, int vpadding, int hpadding,
                                 int vstride, int hstride,
                                 int vdilation, int hdilation):

    cdef const npDT[:,:] rows_view = rows
    cdef npDT[:,:,:,:] x_view = x

    _row2im_1ch_nhwc_cython_inner(rows_view, x_view, n, h, w, c, hh, ww, kh, kw, hpadding, vpadding,
                                  vstride, hstride, vdilation, hdilation)
# --- END row2im_1ch_nhwc_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _row2im_1ch_nhwc_cython_inner(const npDT[:,:] rows,
                                   npDT[:,:,:,:] x,
                                   int n, int h, int w, int c, int hh, int ww,
                                   int kh, int kw, int vpadding, int hpadding,
                                   int vstride, int hstride,
                                   int vdilation, int hdilation):
    cdef int nn, xx, yy, row, cc, ii, jj, col, x_x, x_y

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    row = nn * hh * ww * c + xx * ww * c + yy * c + cc
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = cc * kh * kw + ii * kw + jj
                                    x[nn, x_x, x_y, cc] += rows[row, col]
# --- END row2im_1ch_nhwc_cython_inner --- #

# --- END row2im --- #
