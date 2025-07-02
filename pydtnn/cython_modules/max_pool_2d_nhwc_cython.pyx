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

__all__ = (
    "max_pool_2d_fwd_nhwc_cython",
    "max_pool_2d_bwd_nhwc_cython",
)

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def max_pool_2d_fwd_nhwc_cython(npDT[:,:,:,::1] x,
                                npDT[:,:,:,::1] y,
                                np.int32_t[:,:,:,::1] idx_max,
                                int kh, int kw, int ho, int wo,
                                int vpadding, int hpadding,
                                int vstride, int hstride, 
                                int vdilation, int hdilation,
                                npDT minval) -> None:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, idx_maxval
    cdef npDT maxval, val

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
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
                    y[nn, xx, yy, cc] = maxval
                    idx_max[nn, xx, yy, cc] = idx_maxval
                    
# --- END max_pool_2d_fwd_nhwc_cython --- #
# --- END Forward --- #


# =================== #

# =================== #


# --- Backward --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def max_pool_2d_bwd_nhwc_cython(npDT[:,:,:,::1] dy,
                                np.int32_t[:,:,:,::1] idx_max,
                                npDT[:,:,:,::1] dx,
                                int n, int h, int w, int c,
                                int kh, int kw, int ho, int wo,
                                int vpadding, int hpadding,
                                int vstride, int hstride,
                                int vdilation, int hdilation) -> None:

    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, idx_maxval

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    idx_maxval = idx_max[nn, xx, yy, cc]
                    ii, jj = idx_maxval // kh, idx_maxval % kw
                    x_x = vstride * xx + vdilation * ii - vpadding
                    x_y = hstride * yy + hdilation * jj - hpadding
                    if 0 <= x_x < h and 0 <= x_y < w:
                        dx[nn, x_x, x_y, cc] += dy[nn, xx, yy, cc]
# --- END max_pool_2d_bwd_nhwc_cython --- #
# --- END Backward --- #
