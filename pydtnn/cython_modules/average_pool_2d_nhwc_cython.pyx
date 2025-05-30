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

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #

# --- END COMMON --- #

# =================== #
# =================== #

# --- FORWARD --- #

def average_pool_2d_fwd_nhwc_cython(np.ndarray[npDT, ndim=4] x, 
                                    int kh, int kw, int vpadding, int hpadding,
                                    int vstride, int hstride, int vdilation, int hdilation) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef np.ndarray[npDT, ndim=4] y = np.empty((n, hh, ww, c), dtype=x.dtype)

    try:
        average_pool_2d_fwd_nhwc_cython_inner(y, x, n, h, w, c,
                                            hh, ww, kh, kw, vpadding, hpadding,
                                            vstride, hstride, vdilation, hdilation)
    except TypeError as e:
        raise TypeError(f"Function: \"average_pool_2d_fwd_nhwc_cython\". Error: {e}")

    return y
# --- END average_pool_2d_fwd_nhwc_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int average_pool_2d_fwd_nhwc_cython_inner(np.ndarray[npDT, ndim=4] y,
                                               np.ndarray[npDT, ndim=4] x,
                                               int n, int h, int w, int c, int hh, int ww,
                                               int kh, int kw, int vpadding, int hpadding,
                                               int vstride, int hstride,
                                               int vdilation, int hdilation):
    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, items
    cdef npDT accum

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    accum, items = 0, 0
                    # accum, items = 0, (kh * kw)
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    accum = accum + x[nn, x_x, x_y, cc]
                                    items = items + 1
                    y[nn, xx, yy, cc] = accum // items
# --- END average_pool_2d_fwd_nhwc_cython_inner --- #

# --- END FORWARD --- #

# =================== #
# =================== #

# --- BACKWARD --- #

def average_pool_2d_bwd_nhwc_cython(np.ndarray[npDT, ndim=4] y,
                                    int n, int h, int w, int c,
                                    int kh, int kw,
                                    int vpadding, int hpadding,
                                    int vstride, int hstride,
                                    int vdilation, int hdilation) -> np.ndarray:

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef np.ndarray x = np.empty((n, h, w, c), dtype=y.dtype)
    
    try:
        average_pool_2d_bwd_nhwc_cython_inner(y, x, 
                                              n, h, w, c,
                                              hh, ww, kh, kw, 
                                              vpadding, hpadding,
                                              vstride, hstride, 
                                              vdilation, hdilation)
    except TypeError as e:
        raise TypeError(f"Function: \"average_pool_2d_bwd_nhwc_cython\". Error: {e}")

    return x
# --- END average_pool_2d_bwd_nhwc_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int average_pool_2d_bwd_nhwc_cython_inner(np.ndarray[npDT, ndim=4] y,
                                               np.ndarray[npDT, ndim=4] x,
                                               int n, int h, int w, int c, 
                                               int hh, int ww, int kh, int kw, 
                                               int vpadding, int hpadding,
                                               int vstride, int hstride,
                                               int vdilation, int hdilation):

    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, items
    cdef npDT avgval

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    items = 0
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    items = items + 1
                    avgval = y[nn, xx, yy, cc] // items
                    # avgval = y[nn, xx, yy, cc] // (kh * kw)
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    x[nn, x_x, x_y, cc] += avgval
# --- END average_pool_2d_bwd_nhwc_cython_inner --- #

# --- END BACKWARD --- #
