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

def bn_relu_inference_cython(x, running_mean, inv_std, gamma, beta):
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta
    shape = x.shape

    cdef np.ndarray y = np.zeros((shape,), dtype=x.dtype, order="F")

    if x.dtype == np.int8:
        bn_relu_inference_cython_inner_int8(x, running_mean, inv_std, y, gamma, beta)
    elif x.dtype == np.float32:
        bn_relu_inference_cython_inner_float32(x, running_mean, inv_std, y, gamma, beta)
    elif x.dtype == np.float64:
        bn_relu_inference_cython_inner_float64(x, running_mean, inv_std, y, gamma, beta)
    else:
        raise TypeError("Type {str(x.dtype)} is not supported for bn_relu_inference_cython!")

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_relu_inference_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] x,
                                         np.ndarray[np.int8_t, ndim=1] running_mean,
                                         np.ndarray[np.int8_t, ndim=1] inv_std,
                                         np.ndarray[np.int8_t, ndim=2] y,
                                         np.ndarray[np.int8_t, ndim=1] gamma,
                                         np.ndarray[np.int8_t, ndim=1] beta):
    cdef int i, j = 0
    cdef int tmp
    for j in prange(x.shape[1], nogil=True, schedule='static'):
        for i in range(x.shape[0]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = max((tmp * gamma[j]) + beta[j], 0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_relu_inference_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] x,
                                            np.ndarray[np.float32_t, ndim=1] running_mean,
                                            np.ndarray[np.float32_t, ndim=1] inv_std,
                                            np.ndarray[np.float32_t, ndim=2] y,
                                            np.ndarray[np.float32_t, ndim=1] gamma,
                                            np.ndarray[np.float32_t, ndim=1] beta):
    cdef int i, j
    cdef float tmp
    for j in prange(x.shape[1], nogil=True, schedule='static'):
        for i in range(x.shape[0]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = max((tmp * gamma[j]) + beta[j], 0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_relu_inference_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] x,
                                            np.ndarray[np.float64_t, ndim=1] running_mean,
                                            np.ndarray[np.float64_t, ndim=1] inv_std,
                                            np.ndarray[np.float64_t, ndim=2] y,
                                            np.ndarray[np.float64_t, ndim=1] gamma,
                                            np.ndarray[np.float64_t, ndim=1] beta):
    cdef int i, j
    cdef double tmp
    for j in prange(x.shape[1], nogil=True, schedule='static'):
        for i in range(x.shape[0]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = max((tmp * gamma[j]) + beta[j], 0)
