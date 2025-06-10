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
def bn_training_fwd_cython(x: np.ndarray, 
                           gamma: np.ndarray, 
                           beta: np.ndarray, 
                           running_mean: np.ndarray, 
                           running_var: np.ndarray, 
                           float momentum, 
                           float eps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    y:np.ndarray   = np.empty_like(x, dtype=x.dtype, order="C")
    xn:np.ndarray  = np.empty_like(x, dtype=x.dtype, order="C")
    xc:np.ndarray  = np.empty_like(x, dtype=x.dtype, order="C")
    std:np.ndarray = np.empty((x.shape[1],), dtype=x.dtype)

    try:
        bn_training_fwd_cython_inner(x, y, xn, xc, std, gamma, beta, running_mean, running_var, momentum, eps)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_training_fwd_cython\". Error: {e}")

    return y, std, xn
# --- END bn_training_fwd_cython --- #


def bn_training_fwd_cython_inner(np.ndarray[npDT, ndim=2] x,
                                 np.ndarray[npDT, ndim=2] y,
                                 np.ndarray[npDT, ndim=2] xn,
                                 np.ndarray[npDT, ndim=2] xc,
                                 np.ndarray[npDT, ndim=1] std,
                                 np.ndarray[npDT, ndim=1] gamma,
                                 np.ndarray[npDT, ndim=1] beta,
                                 np.ndarray[npDT, ndim=1] running_mean,
                                 np.ndarray[npDT, ndim=1] running_var,
                                 float momentum, 
                                 float eps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    cdef npDT[:,:] x_view = x
    cdef npDT[:,:] y_view = y
    cdef npDT[:,:] xn_view = xn
    cdef npDT[:,:] xc_view = xc
    cdef npDT[:] std_view = std
    cdef const npDT[:] gamma_view = gamma
    cdef const npDT[:] beta_view = beta
    cdef npDT[:] running_mean_view = running_mean
    cdef npDT[:] running_var_view = running_var

    _bn_training_fwd_cython_inner(x_view, y_view, xn_view, xc_view, std_view, gamma_view, beta_view, running_mean_view, running_var_view, momentum, eps)

# --- END bn_training_fwd_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _bn_training_fwd_cython_inner(npDT[:,:] x, npDT[:,:] y,
                                   npDT[:,:] xn, npDT[:,:] xc,
                                   npDT[:] std,
                                   const npDT[:] gamma, const npDT[:] beta,
                                   npDT[:] running_mean, npDT[:] running_var,
                                   float momentum, float eps):
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
# --- bn_training_fwd_cython_inner --- #


# --- END FORWARD --- #



# --- BACKWARD --- #
def bn_training_bwd_cython(dy: np.ndarray,
                           xn: np.ndarray,
                           std: np.ndarray,
                           gamma: np.ndarray,
                           dgamma: np.ndarray,
                           dbeta: np.ndarray) -> np.ndarray:

    dx:np.ndarray  = np.empty_like(dy, dtype=dy.dtype, order="C")

    try:
        _bn_training_bwd_cython_inner(dx, dy, xn, std, gamma, dgamma, dbeta)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_training_bwd_cython\". Error: {e}")

    return dx
# --- bn_training_bwd_cython --- #

def _bn_training_bwd_cython_inner(np.ndarray[npDT, ndim=2] dx,
                                  np.ndarray[npDT, ndim=2] dy,
                                  np.ndarray[npDT, ndim=2] xn,
                                  np.ndarray[npDT, ndim=1] std,
                                  np.ndarray[npDT, ndim=1] gamma,
                                  np.ndarray[npDT, ndim=1] dgamma,
                                  np.ndarray[npDT, ndim=1] dbeta):
    cdef npDT[:,:] dx_view = dx
    cdef const npDT[:,:] dy_view = dy
    cdef const npDT[:,:] xn_view = xn
    cdef const npDT[:] std_view = std
    cdef const npDT[:] gamma_view = gamma
    cdef const npDT[:] dgamma_view = dgamma
    cdef const npDT[:] dbeta_view = dbeta    

    bn_training_bwd_cython_inner(dx_view, dy_view, xn_view, std_view, gamma_view, dgamma_view, dbeta_view)
# --- _bn_training_bwd_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_training_bwd_cython_inner(npDT[:,:] dx,
                                  const npDT[:,:] dy,
                                  const npDT[:,:] xn,
                                  const npDT[:] std,
                                  const npDT[:] gamma,
                                  const npDT[:] dgamma,
                                  const npDT[:] dbeta):
    cdef int i, j, n = dy.shape[0]

    for i in prange(dy.shape[0], nogil=True, schedule='static'):
        for j in range(dy.shape[1]):
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta) 
            dx[i, j] = (gamma[j] // (std[j] * n)) * (n * dy[i, j] - xn[i, j] * dgamma[j] - dbeta[j])
#--- END bn_training_bwd_cython_inner --- #

# --- END BACKWARD --- #
