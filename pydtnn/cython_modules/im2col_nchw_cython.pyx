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
    "im2col_1ch_nchw_cython",
    "col2im_1ch_nchw_cython",
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

# --- im2col --- #

# NOTE:
# This code has been inspired from cthorey, see:
#    https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2col_cython.pyx
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2col_nchw_cython(npDT[:,:,:,::1] x,
                       npDT[:,::1] cols,
                       int kh, int kw, int ho, int wo,
                       int vpadding, int hpadding,
                       int vstride, int hstride, int vdilation, int hdilation) -> None:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * ho * wo + xx * wo + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]
# --- im2col_nchw_cython --- #
# --- END im2col --- #

# ================== #

# ================== #

# --- col2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def col2im_nchw_cython(npDT[:,::1] cols,
                       npDT[:,:,:,::1] dx, 
                       int n, int c, int h, int w,
                       int kh, int kw, int ho, int wo, 
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation) -> None:

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * ho * wo + xx * wo + yy
                                    dx[nn, cc, x_x, x_y] += cols[row, col]
# --- END col2im_nchw_cython --- #

# ================================== #

# ================================== #


# ---- im2col_nchw_3x3_cython_inner ---- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef im2col_nchw_3x3_cython_inner(npDT[:,::1] cols,
                                  npDT[:,:,:,::1] x,
                                  int n, int c, int h, int w, int ho, int wo,
                                  int kh, int kw, int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation):
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True, schedule='static'):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(ho):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(wo):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * ho * wo + xx * wo + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]
# --- END im2col_nchw_3x3_cython_inner --- #
