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

def relu_cython(x):
    shape = x.shape
    cdef np.ndarray max = np.zeros((np.prod(shape)), dtype=x.dtype)
    cdef np.ndarray mask = np.zeros((np.prod(shape)), dtype=np.int8)

    if x.dtype == np.int8:
        relu_cython_inner_int8(x.reshape(-1), max, mask)
    elif x.dtype == np.float32:
        relu_cython_inner_float32(x.reshape(-1), max, mask)
    elif x.dtype == np.float64:
        relu_cython_inner_float64(x.reshape(-1), max, mask)
    else:
        raise TypeError("Type '{}' is not supported by relu_cython!" % (str(x.dtype)))

    return max.reshape(shape), mask.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner_int8(np.ndarray[np.int8_t, ndim=1] x,
                            np.ndarray[np.int8_t, ndim=1] max,
                            np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i, j = 0
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner_float32(np.ndarray[np.float32_t, ndim=1] x,
                               np.ndarray[np.float32_t, ndim=1] max,
                               np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner_float64(np.ndarray[np.float64_t, ndim=1] x,
                               np.ndarray[np.float64_t, ndim=1] max,
                               np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
