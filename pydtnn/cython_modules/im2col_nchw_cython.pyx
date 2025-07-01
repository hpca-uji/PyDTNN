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
                       int kh, int kw, int vpadding, int hpadding,
                       int vstride, int hstride, int vdilation, int hdilation) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef npDT[:,::1] cols = np.zeros((c * kh * kw, n * hh * ww))

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * hh * ww + xx * ww + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]
    return cols
# --- im2col_nchw_cython --- #
# --- END im2col --- #

# ================== #

# ================== #

# --- col2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def col2im_nchw_cython(npDT[:,::1] cols,
                       int n, int c, int h, int w,
                       int kh, int kw,
                       int vpadding, int hpadding,
                       int vstride, int hstride,
                       int vdilation, int hdilation) -> np.ndarray:

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef npDT[:,:,:,::1] x = np.zeros((n, c, h, w))

    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * hh * ww + xx * ww + yy
                                    x[nn, cc, x_x, x_y] += cols[row, col]

    return x
# --- END col2im_nchw_cython --- #

#                                   x_x                           x_y
#                           x[n, c, vstride * xx + vdilation * ii - vpadding, hstride * yy + hdilation * jj - hpadding] += cols[]
# Throw away 1)
# x_x = vstride * xx + vdilation * ii - vpadding
# if x_x < 0 or x_x >= H:
#   continue
#
# Throw away 2)
# x_y = hstride * yy + hdilation * jj - hpadding
# if x_y < 0 or x_y >= W:
#  continue

# Alternative to throw away 1)
# Range for xx: from:  / a >=0
#                      \ vstride * xx + vdilation * ii - vpadding >= 0
#                         -> a >= (vpadding - ii) // vstride
#                      -> xx = max(0, (vpadding - ii) // vstride))

#               to:    / xx < HH
#                      \ vstride * xx + vdilation * ii - vpadding < H
#                         -> xx < H + (vpadding - ii) // vstride
#                      -> xx = min(HH, H + (vpadding - ii) // vstride))

# --- END col2im --- #

# ================================== #

# ================================== #


# ---- im2col_nchw_3x3_cython_inner ---- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef im2col_nchw_3x3_cython_inner(npDT[:,::1] cols,
                                  npDT[:,:,:,::1] x,
                                  int n, int c, int h, int w, int hh, int ww,
                                  int kh, int kw, int vpadding, int hpadding,
                                  int vstride, int hstride,
                                  int vdilation, int hdilation):
    cdef int cc, ii, jj, row, yy, xx, nn, col, x_x, x_y

    for cc in prange(c, nogil=True, schedule='static'):
        for ii in range(kh):
            for jj in range(kw):
                row = cc * kh * kw + ii * kw + jj
                for nn in range(n):
                    for xx in range(hh):
                        x_x = vstride * xx + vdilation * ii - vpadding
                        if 0 <= x_x < h:
                            for yy in range(ww):
                                x_y = hstride * yy + hdilation * jj - hpadding
                                if 0 <= x_y < w:
                                    col = nn * hh * ww + xx * ww + yy
                                    cols[row, col] = x[nn, cc, x_x, x_y]
# --- END im2col_nchw_3x3_cython_inner --- #
