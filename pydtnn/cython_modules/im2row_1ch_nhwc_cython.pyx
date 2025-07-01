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
    "im2row_1ch_nhwc_cython",
    "row2im_1ch_nhwc_cython",
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

# --- im2row --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def im2row_1ch_nhwc_cython(npDT[:,:,:,::1] x,
                           int kh, int kw, int vpadding, int hpadding,
                           int vstride, int hstride, int vdilation, int hdilation) -> np.ndarray:
    cdef int n = x.shape[0]
    cdef int h = x.shape[1]
    cdef int w = x.shape[2]
    cdef int c = x.shape[3]

    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef npDT[:,::1] rows = np.zeros((n * c * hh * ww, kh * kw))

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

    return rows
# --- END im2row --- #


# ================== #

# ================== #


# --- row2im --- #
@cython.boundscheck(False)
@cython.wraparound(False)
def row2im_1ch_nhwc_cython(npDT[:,::1] rows,
                           int n, int h, int w, int c,
                           int kh, int kw,
                           int vpadding, int hpadding,
                           int vstride, int hstride,
                           int vdilation, int hdilation) -> np.ndarray:
    cdef int hh = (h + 2 * vpadding - vdilation * (kh - 1) - 1) // vstride + 1
    cdef int ww = (w + 2 * hpadding - hdilation * (kw - 1) - 1) // hstride + 1

    cdef npDT[:,:,:,::1] x = np.zeros((n, h, w, c))

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

    return x
# --- END row2im_1ch_nhwc_cython --- #

# --- END row2im --- #
