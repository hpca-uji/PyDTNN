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
from libc.math cimport sqrt

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #

# =================== #
# =================== #

# --- FORWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_training_fwd_cython(np.ndarray[npDT, ndim=2] x,
                           npDT[::1] gamma,
                           npDT[::1] beta,
                           npDT[::1] running_mean,
                           npDT[::1] running_var,
                           float momentum,
                           float eps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    cdef np.ndarray[npDT, ndim=2] y   = np.zeros_like(x, dtype=x.dtype, order="C")
    cdef np.ndarray[npDT, ndim=2] xn  = np.zeros_like(x, dtype=x.dtype, order="C")
    cdef np.ndarray[npDT, ndim=2] xc  = np.zeros_like(x, dtype=x.dtype, order="C")
    cdef np.ndarray[npDT, ndim=1] std = np.zeros((x.shape[1],), dtype=x.dtype)

    cdef int i, j
    cdef npDT mu, var

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
        std[j] = <npDT> (sqrt(var + eps))

        # self.xn = xc / self.std
        # y = self.gamma * self.xn + self.beta
        for i in range(x.shape[0]):
            xn[i, j] = xc[i, j] // std[j]
            y[i, j] = gamma[j] * xn[i, j] + beta[j]

        # self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
        # self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        running_mean[j] = <npDT> (momentum * running_mean[j] + (1.0 - momentum) * mu)
        running_var[j] = <npDT> (momentum * running_var[j] + (1.0 - momentum) * var)

    return y, std, xn
# --- END bn_training_fwd_cython --- #

# --- END FORWARD --- #

# --- BACKWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_training_bwd_cython(npDT[:, ::1] dx,
                           npDT[:, ::1] dy,
                           npDT[:, ::1] xn,
                           npDT[::1] std,
                           npDT[::1] gamma,
                           npDT[::1] dgamma,
                           npDT[::1] dbeta) -> None:

    cdef int i, j, n = dy.shape[0]

    for i in prange(n, nogil=True, schedule='static'):
        for j in range(dy.shape[1]):
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta) 
            dx[i, j] = (<npDT> (gamma[j] / (std[j] * n)) * (n * dy[i, j] - xn[i, j] * dgamma[j] - dbeta[j]))
# --- bn_training_bwd_cython --- #

# --- END BACKWARD --- #
