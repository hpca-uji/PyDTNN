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
from libc.math cimport sqrt

def bn_training_fwd_cython(x, gamma, beta, running_mean, running_var, momentum, eps):

    cdef np.ndarray y   = np.zeros_like(x, dtype=x.dtype, order="C")
    cdef np.ndarray xn  = np.zeros_like(x, dtype=x.dtype, order="C")
    cdef np.ndarray xc  = np.zeros_like(x, dtype=x.dtype, order="C")
    cdef np.ndarray std = np.zeros((x.shape[1],), dtype=x.dtype)

    if x.dtype == np.int8:
        bn_training_fwd_cython_inner_int8(x, y, xn, xc, std, gamma, beta, \
                                         running_mean, running_var, momentum, eps)
    elif x.dtype == np.float32:
        bn_training_fwd_cython_inner_float32(x, y, xn, xc, std, gamma, beta, \
                                         running_mean, running_var, momentum, eps)
    elif x.dtype == np.float64:
        bn_training_fwd_cython_inner_float64(x, y, xn, xc, std, gamma, beta, \
                                         running_mean, running_var, momentum, eps)
    else:
        raise TypeError(f"Type {str(x.dtype)} is not supported by bn_training_fwd_cython!")

    return y, std, xn

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_fwd_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] x,
                                    np.ndarray[np.int8_t, ndim=2] y,
                                    np.ndarray[np.int8_t, ndim=2] xn,
                                    np.ndarray[np.int8_t, ndim=2] xc,
                                    np.ndarray[np.int8_t, ndim=1] std,
                                    np.ndarray[np.int8_t, ndim=1] gamma,
                                    np.ndarray[np.int8_t, ndim=1] beta,
                                    np.ndarray[np.int8_t, ndim=1] running_mean,
                                    np.ndarray[np.int8_t, ndim=1] running_var,
                                    float momentum,
                                    float eps):
    cdef int i, j
    cdef np.int8_t mu, var

    for j in prange(x.shape[1], nogil=True, schedule='static'):
        # mu = mean(x, n, self.model.comm)
        mu = 0
        for i in range(x.shape[0]):
            mu += x[i, j]
        mu = mu // x.shape[0]

        # xc = (x - mu)
        # var = mean(xc ** 2, n, self.model.comm)
        var = 0
        for i in range(x.shape[0]):
            xc[i, j] = x[i, j] - mu
            var += xc[i, j] * xc[i, j]
        var = var // x.shape[0]

        # self.std = np.sqrt(var + self.epsilon)
        std[j] = int(sqrt(var + eps))

        # self.xn = xc / self.std
        # y = self.gamma * self.xn + self.beta
        for i in range(x.shape[0]):
            xn[i, j] = xc[i, j] // std[j]
            y[i, j] = gamma[j] * xn[i, j] + beta[j]

        # self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
        # self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        running_mean[j] = int(momentum * running_mean[j] + (1.0 - momentum) * mu)
        running_var[j] = int(momentum * running_var[j] + (1.0 - momentum) * var)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_fwd_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] x,
                                    np.ndarray[np.float32_t, ndim=2] y,
                                    np.ndarray[np.float32_t, ndim=2] xn,
                                    np.ndarray[np.float32_t, ndim=2] xc,
                                    np.ndarray[np.float32_t, ndim=1] std,
                                    np.ndarray[np.float32_t, ndim=1] gamma,
                                    np.ndarray[np.float32_t, ndim=1] beta,
                                    np.ndarray[np.float32_t, ndim=1] running_mean,
                                    np.ndarray[np.float32_t, ndim=1] running_var,
                                    float momentum,
                                    float eps):
    cdef int i, j
    cdef np.float32_t mu, var

    for j in prange(x.shape[1], nogil=True, schedule='static'):
        # mu = mean(x, n, self.model.comm)
        mu = 0
        for i in range(x.shape[0]):
            mu += x[i, j]
        mu = mu / x.shape[0]

        # xc = (x - mu)
        # var = mean(xc ** 2, n, self.model.comm)
        var = 0
        for i in range(x.shape[0]): 
            xc[i, j] = x[i, j] - mu
            var += xc[i, j] * xc[i, j]
        var = var / x.shape[0]

        # self.std = np.sqrt(var + self.epsilon)
        std[j] = sqrt(var + eps)

        # self.xn = xc / self.std
        # y = self.gamma * self.xn + self.beta
        for i in range(x.shape[0]): 
            xn[i, j] = xc[i, j] / std[j]
            y[i, j] = gamma[j] * xn[i, j] + beta[j]

        # self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
        # self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        running_mean[j] = momentum * running_mean[j] + (1.0 - momentum) * mu
        running_var[j] = momentum * running_var[j] + (1.0 - momentum) * var

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_fwd_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] x,
                                    np.ndarray[np.float64_t, ndim=2] y,
                                    np.ndarray[np.float64_t, ndim=2] xn,
                                    np.ndarray[np.float64_t, ndim=2] xc,
                                    np.ndarray[np.float64_t, ndim=1] std,
                                    np.ndarray[np.float64_t, ndim=1] gamma,
                                    np.ndarray[np.float64_t, ndim=1] beta,
                                    np.ndarray[np.float64_t, ndim=1] running_mean,
                                    np.ndarray[np.float64_t, ndim=1] running_var,
                                    float momentum,
                                    float eps):
    cdef int i, j
    cdef np.float64_t mu, var

    for j in prange(x.shape[1], nogil=True, schedule='static'):
        # mu = mean(x, n, self.model.comm)
        mu = 0
        for i in range(x.shape[0]):
            mu += x[i, j]
        mu = mu // x.shape[0]

        # xc = (x - mu)
        # var = mean(xc ** 2, n, self.model.comm)
        var = 0
        for i in range(x.shape[0]):
            xc[i, j] = x[i, j] - mu
            var += xc[i, j] * xc[i, j]
        var = var // x.shape[0]

        # self.std = np.sqrt(var + self.epsilon)
        std[j] = sqrt(var + eps)

        # self.xn = xc / self.std
        # y = self.gamma * self.xn + self.beta
        for i in range(x.shape[0]):
            xn[i, j] = xc[i, j] // std[j]
            y[i, j] = gamma[j] * xn[i, j] + beta[j]

        # self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
        # self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        running_mean[j] = momentum * running_mean[j] + (1.0 - momentum) * mu
        running_var[j] = momentum * running_var[j] + (1.0 - momentum) * var

def bn_training_bwd_cython(dy, std, xn, gamma, dgamma, dbeta):

    cdef np.ndarray dx = np.zeros_like(dy, dtype=dy.dtype, order="C")

    if dy.dtype == np.int8:
        bn_training_bwd_cython_inner_int8(dx, dy, xn, std, gamma, dgamma, dbeta)
    elif dy.dtype == np.float32:
        bn_training_bwd_cython_inner_float32(dx, dy, xn, std, gamma, dgamma, dbeta)
    elif dy.dtype == np.float64:
        bn_training_bwd_cython_inner_float64(dx, dy, xn, std, gamma, dgamma, dbeta)
    else:
        raise TypeError(f"Type {str(dy.dtype)} is not supported by bn_training_fwd_cython!")

    return dx

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_bwd_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] dx,
                                    np.ndarray[np.int8_t, ndim=2] dy,
                                    np.ndarray[np.int8_t, ndim=2] xn,
                                    np.ndarray[np.int8_t, ndim=1] std,
                                    np.ndarray[np.int8_t, ndim=1] gamma,
                                    np.ndarray[np.int8_t, ndim=1] dgamma,
                                    np.ndarray[np.int8_t, ndim=1] dbeta):
    cdef int i, j, n = dy.shape[0]

    for i in prange(dy.shape[0], nogil=True, schedule='static'):
        for j in range(dy.shape[1]):
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta) 
            dx[i, j] = (gamma[j] // (std[j] * n)) * (n * dy[i, j] - xn[i, j] * dgamma[j] - dbeta[j])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_bwd_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] dx,
                                    np.ndarray[np.float32_t, ndim=2] dy,
                                    np.ndarray[np.float32_t, ndim=2] xn,
                                    np.ndarray[np.float32_t, ndim=1] std,
                                    np.ndarray[np.float32_t, ndim=1] gamma,
                                    np.ndarray[np.float32_t, ndim=1] dgamma,
                                    np.ndarray[np.float32_t, ndim=1] dbeta):
    cdef int i, j, n = dy.shape[0]

    for i in prange(dy.shape[0], nogil=True, schedule='static'):
        for j in range(dy.shape[1]):
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta) 
            dx[i, j] = (gamma[j] / (std[j] * n)) * (n * dy[i, j] - xn[i, j] * dgamma[j] - dbeta[j])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_bwd_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] dx,
                                    np.ndarray[np.float64_t, ndim=2] dy,
                                    np.ndarray[np.float64_t, ndim=2] xn,
                                    np.ndarray[np.float64_t, ndim=1] std,
                                    np.ndarray[np.float64_t, ndim=1] gamma,
                                    np.ndarray[np.float64_t, ndim=1] dgamma,
                                    np.ndarray[np.float64_t, ndim=1] dbeta):
    cdef int i, j, n = dy.shape[0]

    for i in prange(dy.shape[0], nogil=True, schedule='static'):
        for j in range(dy.shape[1]):
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta) 
            dx[i, j] = (gamma[j] / (std[j] * n)) * (n * dy[i, j] - xn[i, j] * dgamma[j] - dbeta[j])
