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

def pointwise_conv_cython(x: np.ndarray, k: np.ndarray) -> np.ndarray:

    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int co = k.shape[0]

    out: np.ndarray = np.empty((n, co, h, w), dtype=x.dtype)

    try:
        pointwise_conv_cython_inner(out, x, k, n, c, h, w, co)
    except TypeError as e:
        raise TypeError(f"Function: \"pointwise_conv_cython\". Error: {e}")

    return out
# --- END pointwise_conv_cython --- #

def pointwise_conv_cython_inner(np.ndarray[npDT, ndim=4] out,
                                np.ndarray[npDT, ndim=4] x,
                                np.ndarray[npDT, ndim=2] k,
                                int n, int c, int h, int w, int co):

    cdef npDT[:,:,:,:] out_view = out
    cdef const npDT[:,:,:,:] x_view = x
    cdef const npDT[:,:] k_view = k

    _pointwise_conv_cython_inner(out_view, x_view, k_view, n, c, h, w, co)
# --- END pointwise_conv_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _pointwise_conv_cython_inner(npDT[:,:,:,:] out,
                                  const npDT[:,:,:,:] x,
                                  const npDT[:,:] k,
                                  int n, int c, int h, int w, int co):
    cdef int nn, cco, cc, ii, jj

    for cco in prange(co, nogil=True):
        for cc in range(c):
            for nn in range(n):
                for ii in range(h):
                    for jj in range(w):
                        out[nn, cco, ii, jj] += x[nn, cc, ii, jj] * k[cco, cc]
# --- END _pointwise_conv_cython_inner --- #
