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

def max_pool_2d_fwd_nhwc_cython(x: np.ndarray, 
                                int kh, int kw, 
                                int vpadding, int hpadding,
                                int vstride, int hstride, 
                                int vdilation, int hdilation) -> tuple[np.ndarray, np.ndarray]:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    y: np.ndarray = np.empty((n, hh, ww, c), dtype=x.dtype)
    idx_max: np.ndarray = np.empty((n, hh, ww, c), dtype=np.int32)    

    try:
        max_pool_2d_fwd_nhwc_cython_inner(y, x, idx_max, n, h, w, c,
                                          hh, ww, kh, kw, vpadding, hpadding,
                                          vstride, hstride, vdilation, hdilation)
    except TypeError as e:
        raise TypeError(f"Function: \"max_pool_2d_fwd_nchw_cython\". Error: {e}")

    return y, idx_max
# --- END max_pool_2d_fwd_nhwc_cython --- #

def max_pool_2d_fwd_nhwc_cython_inner(np.ndarray[npDT, ndim=4] y,
                                      np.ndarray[npDT, ndim=4] x,
                                      np.ndarray[np.int32_t, ndim=4] idx_max,
                                      int n, int h, int w, int c, int hh, int ww,
                                      int kh, int kw, int vpadding, int hpadding,
                                      int vstride, int hstride,
                                      int vdilation, int hdilation):

    cdef npDT[:,:,:,:] y_view = y
    cdef const npDT[:,:,:,:] x_view = x
    cdef npDT minval = np.iinfo(x.dtype).min if np.issubdtype(x.dtype, np.integer) else np.finfo(x.dtype).min 

    _max_pool_2d_fwd_nhwc_cython_inner(y_view, x_view, idx_max, n, h, w, c,
                                       hh, ww, kh, kw, vpadding, hpadding,
                                       vstride, hstride, vdilation, hdilation, minval)

# --- END max_pool_2d_fwd_nhwc_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _max_pool_2d_fwd_nhwc_cython_inner(npDT[:,:,:,:] y,
                                        const npDT[:,:,:,:] x,
                                        np.ndarray[np.int32_t, ndim=4] idx_max,
                                        int n, int h, int w, int c, int hh, int ww,
                                        int kh, int kw, int vpadding, int hpadding,
                                        int vstride, int hstride,
                                        int vdilation, int hdilation, 
                                        npDT minval):

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef npDT maxval, val

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    maxval, idx_maxval = minval, 0
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    val = x[nn, x_x, x_y, cc]
                                    if val > maxval:
                                        maxval, idx_maxval = val, ii * kw + jj
                    y[nn, xx, yy, cc], idx_max[nn, xx, yy, cc] = maxval, idx_maxval
# --- END _max_pool_2d_fwd_nhwc_cython_inner --- #

# --- END Forward --- #


# =================== #

# =================== #


# --- Backward --- #

def max_pool_2d_bwd_nhwc_cython(y: np.ndarray, 
                                np.ndarray[np.int32_t, ndim=4] idx_max,
                                int n, int h, int w, int c,
                                int kh, int kw,
                                int vpadding, int hpadding,
                                int vstride, int hstride,
                                int vdilation, int hdilation) -> np.ndarray:

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    x: np.ndarray = np.empty((n, h, w, c), dtype=y.dtype)
    print(f"{y.dtype=}")

    try:
        max_pool_2d_bwd_nhwc_cython_inner(y, x, idx_max, n, h, w, c,
                                          hh, ww, kh, kw, 
                                          vpadding, hpadding,
                                          vstride, hstride, 
                                          vdilation, hdilation)
    except TypeError as e:
        raise TypeError(f"Function: \"max_pool_2d_bwd_nhwc_cython\". Error: {e}")

    return x
# --- END max_pool_2d_bwd_nhwc_cython --- #

def max_pool_2d_bwd_nhwc_cython_inner(np.ndarray[npDT, ndim=4] y,
                                      np.ndarray[npDT, ndim=4] x,
                                      np.ndarray[np.int32_t, ndim=4] idx_max,
                                      int n, int h, int w, int c, 
                                      int hh, int ww, int kh, int kw, 
                                      int vpadding, int hpadding,
                                      int vstride, int hstride,
                                      int vdilation, int hdilation):

    cdef const npDT[:,:,:,:] y_view = y
    cdef npDT[:,:,:,:] x_view = x

    _max_pool_2d_bwd_nhwc_cython_inner(y_view, x_view, idx_max, n, h, w, c,
                                      hh, ww, kh, kw, 
                                      vpadding, hpadding,
                                      vstride, hstride, 
                                      vdilation, hdilation)
# --- max_pool_2d_bwd_nhwc_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _max_pool_2d_bwd_nhwc_cython_inner(const npDT[:,:,:,:] y,
                                        npDT[:,:,:,:] x,
                                        np.ndarray[np.int32_t, ndim=4] idx_max,
                                        int n, int h, int w, int c, 
                                        int hh, int ww, int kh, int kw, 
                                        int vpadding, int hpadding,
                                        int vstride, int hstride,
                                        int vdilation, int hdilation):
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for xx in range(hh):
            for yy in range(ww):
                for cc in range(c):
                    idx_maxval = idx_max[nn, xx, yy, cc]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = vstride * xx + vdilation * ii - vpadding
                    x_y = hstride * yy + hdilation * jj - hpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        x[nn, x_x, x_y, cc] += y[nn, xx, yy, cc]
# --- _max_pool_2d_bwd_nhwc_cython_inner --- #

# --- END Backward --- #
