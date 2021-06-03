#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

def add_nchw_cython(x, b):
    # if axis == 0:
    #     x = x.T
    # if not x.flags['C_CONTIGUOUS']:
    #     np.ascontiguousarray(x, dtype=np.float32)

    if x.dtype == np.int8:
        add_nchw_cython_inner_int8(x, b)
    elif x.dtype == np.float32:
        add_nchw_cython_inner_float32(x, b)
    elif x.dtype == np.float64:
        add_nchw_cython_inner_float64(x, b)
    else:
        raise TypeError("Type '{}' is not supported by add_nchw_cython!".format(str(x.dtype)))

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef add_nchw_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] x,
                                np.ndarray[np.int8_t, ndim=1] b):
    cdef int i, j
    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef add_nchw_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] x,
                                   np.ndarray[np.float32_t, ndim=1] b):
    cdef int i, j
    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef add_nchw_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] x,
                                   np.ndarray[np.float64_t, ndim=1] b):
    cdef int i, j
    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[i]
