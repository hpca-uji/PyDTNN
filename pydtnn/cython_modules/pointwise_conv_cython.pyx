#
#  This file is part of Python Distributed Training of neural networks (PyDTnn)
#
#  copyright (c) 2021 Universitat Jaume I
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

def pointwise_conv_cython(x, k):
    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int co = k.shape[0]

    cdef np.ndarray out = np.zeros((n, co, h, w), dtype=x.dtype)

    if (x.dtype == np.int8):
        pointwise_conv_cython_inner_int8(out, x, k, n, c, h, w, co)
    elif (x.dtype == np.float32):
        pointwise_conv_cython_inner_float32(out, x, k, n, c, h, w, co)
    elif (x.dtype == np.float64):
        pointwise_conv_cython_inner_float64(out, x, k, n, c, h, w, co)
    else:
        raise TypeError("Type '{}' is not by pointwise_conv_cython!".format(str(out.dtype)))

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int pointwise_conv_cython_inner_int8(np.ndarray[np.int8_t, ndim=4] out,
                                         np.ndarray[np.int8_t, ndim=4] x, 
                                         np.ndarray[np.int8_t, ndim=2] k,
                                         int n, int c, int h, int w, int co):
    cdef int nn, cco, cc, ii, jj

    for cco in prange(co, nogil=True):
        for cc in range(c):
            for nn in range(n):
                for ii in range(h):
                    for jj in range(w):
                        out[nn, cco, ii, jj] += x[nn, cc, ii, jj] * k[cco, cc]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int pointwise_conv_cython_inner_float32(np.ndarray[np.float32_t, ndim=4] out,
                                         np.ndarray[np.float32_t, ndim=4] x, 
                                         np.ndarray[np.float32_t, ndim=2] k,
                                         int n, int c, int h, int w, int co):
    cdef int nn, cco, cc, ii, jj

    for cco in prange(co, nogil=True):
        for cc in range(c):
            for nn in range(n):
                for ii in range(h):
                    for jj in range(w):
                        out[nn, co, ii, jj] += x[nn, cc, ii, jj] * k[cco, cc]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int pointwise_conv_cython_inner_float64(np.ndarray[np.float64_t, ndim=4] out,
                                         np.ndarray[np.float64_t, ndim=4] x, 
                                         np.ndarray[np.float64_t, ndim=2] k,
                                         int n, int c, int h, int w, int co):
    cdef int nn, cco, cc, ii, jj

    for cco in prange(co, nogil=True):
        for cc in range(c):
            for nn in range(n):
                for ii in range(h):
                    for jj in range(w):
                        out[nn, cco, ii, jj] += x[nn, cc, ii, jj] * k[cco, cc]
