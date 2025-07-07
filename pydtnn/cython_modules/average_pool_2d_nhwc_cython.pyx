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
    "average_pool_2d_fwd_nhwc_cython",
    "average_pool_2d_bwd_nhwc_cython"
)

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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_pool_2d_fwd_nhwc_cython(npDT[:,:,:,::1] x,
                                    npDT[:,:,:,::1] y,
                                    int kh, int kw, int ho, int wo,
                                    int vpadding, int hpadding,
                                    int vstride, int hstride, 
                                    int vdilation, int hdilation) -> None:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int cc, ii, jj, yy, xx, nn, x_x, x_y, items
    cdef npDT accum

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    accum = <npDT> 0.0
                    items = 0
                    # accum, items = 0, (kh * kw)
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    accum = accum + x[nn, x_x, x_y, cc]
                                    items = items + 1
                    y[nn, xx, yy, cc] = <npDT> (accum // items)
# --- END average_pool_2d_fwd_nhwc_cython --- #

# --- END FORWARD --- #

# =================== #
# =================== #

# --- BACKWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def average_pool_2d_bwd_nhwc_cython(npDT[:,:,:,::1] dy,
                                    npDT[:,:,:,::1] dx,
                                    int n, int h, int w, int c,
                                    int kh, int kw, int ho, int wo,
                                    int vpadding, int hpadding,
                                    int vstride, int hstride,
                                    int vdilation, int hdilation) -> None:
    
    cdef int nn, xx, yy, cc, ii, jj, x_x, x_y, items
    cdef npDT avgval

    for nn in prange(n, nogil=True):
        for xx in range(ho):
            for yy in range(wo):
                for cc in range(c):
                    items = 0
                    avgval = dy[nn, xx, yy, cc]
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    items = items + 1
                        else: continue
                    avgval /= items
                    # avgval = dy[nn, xx, yy, cc] // (kh * kw)
                    for ii in range(kh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for jj in range(kw):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    dx[nn, x_x, x_y, cc] += avgval
# --- END average_pool_2d_bwd_nhwc_cython --- #

# --- END BACKWARD --- #
