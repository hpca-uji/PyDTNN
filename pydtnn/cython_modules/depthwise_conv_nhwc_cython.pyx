#
#  This file is part of Python Distributed Training of neural networks (PyDTnn)
#
#  copyright (c) 2021-2025 Universitat Jaume I
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

# =============== #
# --- FORWARD --- #

def depthwise_conv_nhwc_cython(x: np.ndarray, k: np.ndarray,
                               int vpadding, int hpadding, 
                               int vstride, int hstride, 
                               int vdilation, int hdilation)-> np.ndarray:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int kh = k.shape[1]
    cdef int kw = k.shape[2]

    cdef int hi = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    res: np.ndarray = np.zeros((n, hi, ww, c), dtype=x.dtype)
    try:
        depthwise_conv_cython_inner(res, x, k, n, c, h, w,
                                    hi, ww, kh, kw, vpadding, hpadding,
                                    vstride, hstride, vdilation, hdilation)
        return res
    except TypeError as e:
        raise TypeError(f"Function: \"depthwise_conv_nhwc_cython\". Error: {e}")
# --- END depthwise_conv_cython --- #

def depthwise_conv_cython_inner(np.ndarray[npDT, ndim=4] res,
                                np.ndarray[npDT, ndim=4] x,
                                np.ndarray[npDT, ndim=3] k,
                                int n, int c, int h, int w, int hi, int ww,
                                int kh, int kw, int vpadding, int hpadding,
                                int vstride, int hstride,
                                int vdilation, int hdilation):

    cdef npDT[:,:,:,:] res_view = res
    cdef const npDT[:,:,:,:] x_view = x
    cdef const npDT[:,:,:] k_view = k

    _depthwise_conv_cython_inner(res_view, x_view, k_view, n, c, h, w,
                                 hi, ww, kh, kw, vpadding, hpadding,
                                 vstride, hstride, vdilation, hdilation)
# --- END depthwise_conv_cython_inner --- #


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _depthwise_conv_cython_inner(npDT[:, :, :, :] res,
                                  const npDT[:, :, :, :] x,
                                  const npDT[:, :, :] k,
                                  int n, int c, int h, int w, int hi, int ww,
                                  int kh, int kw, int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation):

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                for nn in range(n):
                    for xx in range(hi):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    res[nn, xx, yy, cc] += k[cc, ii, jj] * x[nn, x_x, x_y, cc]
# --- END _depthwise_conv_cython_inner --- #

# --- END FORWARD --- #
# =================== #


# =================== #
# ----- BACKWARD ---- #
def depthwise_conv_backward_nhwc_cython(dy: np.ndarray, x: np.ndarray, k: np.ndarray,
                                        int vpadding, int hpadding, 
                                        int vstride, int hstride, 
                                        int vdilation, int hdilation)-> tuple(np.ndarray, np.ndarray):
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int kh = k.shape[1]
    cdef int kw = k.shape[2]

    cdef int hi = dy.shape[1]
    cdef int ww = dy.shape[2]

    dx: np.ndarray = np.zeros((n, h, w, c), dtype=x.dtype)
    dw: np.ndarray = np.zeros((c, kh, kw), dtype=k.dtype)

    try:
        depthwise_conv_backward_inner(dx, dw, dy, x, k, 
                                      n, c, h, w,
                                      hi, ww, kh, kw, 
                                      vpadding, hpadding,
                                      vstride, hstride, 
                                      vdilation, hdilation)
        return dx, dw
    except TypeError as e:
        raise TypeError(f"Function: \"depthwise_conv_backward_nhwc_cython\". Error: {e}")
# --- END depthwise_conv_backward_nhwc_cython --- #

def depthwise_conv_backward_inner(np.ndarray[npDT, ndim=4] dx,
                                  np.ndarray[npDT, ndim=3] dw,
                                  np.ndarray[npDT, ndim=4] dy,
                                  np.ndarray[npDT, ndim=4] x,
                                  np.ndarray[npDT, ndim=3] k,
                                  int n, int c, int h, int w, 
                                  int hi, int ww, int kh, int kw, 
                                  int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation):
    
    cdef npDT[:,:,:,:] dx_view = dx
    cdef npDT[:,:,:] dw_view = dw
    cdef const npDT[:,:,:,:] dy_view = dy
    cdef const npDT[:,:,:,:] x_view = x
    cdef const npDT[:,:,:] k_view = k

    _depthwise_conv_backward_inner(dx_view, dw_view, dy_view, x_view, k_view, 
                                   n, c, h, w,
                                   hi, ww, kh, kw, 
                                   vpadding, hpadding,
                                   vstride, hstride, 
                                   vdilation, hdilation)
# --- END depthwise_conv_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _depthwise_conv_backward_inner(npDT[:, :, :, :] dx,
                                    npDT[:, :, :] dw,
                                    const npDT[:, :, :, :] dy,
                                    const npDT[:, :, :, :] x,
                                    const npDT[:, :, :] k,
                                    int n, int c, int h, int w, 
                                    int hi, int ww, int kh, int kw, 
                                    int vpadding, int hpadding,
                                    int vstride, int hstride,
                                    int vdilation, int hdilation):

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y
    cdef npDT val_k
    
    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):

                val_k = k[cc, ii, jj]
                for nn in range(n):
                    for xx in range(h):

                        x_x = vstride * xx + vdilation * ii - vpadding                        
                        if 0 <= x_x < hi:
                            for yy in range(w):

                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < ww:
                                    
                                    dw[cc, ii, jj] = val_k * x[nn, x_x, x_y, cc]
                                    dx[nn, x_x, x_y, cc] += val_k * dy[nn, xx, yy, cc]
# --- END _depthwise_conv_backward_inner --- #

# --- END FORWARD --- #
# =================== #
