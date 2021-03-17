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

def argmax_cython(x, axis=0):
    if axis == 0: x = x.T
    # if not x.flags['C_CONTIGUOUS']:
    #     np.ascontiguousarray(x, dtype=np.float32)
    cdef np.ndarray max = np.zeros((x.shape[0]), dtype=x.dtype)
    cdef np.ndarray amax = np.zeros((x.shape[0]), dtype=np.int32)
    cdef np.ndarray rng = np.zeros((x.shape[0]), dtype=np.int32)

    if x.dtype == np.int8:
        argmax_cython_inner_int8(x, max, amax, rng)
    elif x.dtype == np.float32:
        argmax_cython_inner_float32(x, max, amax, rng)
    elif x.dtype == np.float64:
        argmax_cython_inner_float64(x, max, amax, rng)
    else:
        raise TypeError("Type '{}' is not supported by argmax_cython!".format(str(x.dtype)))

    return max, tuple([amax, rng])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] x,
                              np.ndarray[np.int8_t, ndim=1] max,
                              np.ndarray[np.int32_t, ndim=1] amax,
                              np.ndarray[np.int32_t, ndim=1] rng):
    cdef int i, j, idx_maxval
    cdef np.int8_t maxval, minval
    minval = np.finfo(np.int8).min

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i, j] > maxval:
                maxval, idx_maxval = x[i, j], j
        amax[i], max[i], rng[i] = idx_maxval, maxval, i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] x,
                                 np.ndarray[np.float32_t, ndim=1] max,
                                 np.ndarray[np.int32_t, ndim=1] amax,
                                 np.ndarray[np.int32_t, ndim=1] rng):
    cdef int i, j, idx_maxval
    cdef np.float32_t maxval, minval
    minval = np.finfo(np.float32).min

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i, j] > maxval:
                maxval, idx_maxval = x[i, j], j
        amax[i], max[i], rng[i] = idx_maxval, maxval, i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] x,
                                 np.ndarray[np.float64_t, ndim=1] max,
                                 np.ndarray[np.int32_t, ndim=1] amax,
                                 np.ndarray[np.int32_t, ndim=1] rng):
    cdef int i, j, idx_maxval
    cdef np.float64_t maxval, minval
    minval = np.finfo(np.float64).min

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i, j] > maxval:
                maxval, idx_maxval = x[i, j], j
        amax[i], max[i], rng[i] = idx_maxval, maxval, i
