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

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #

# --- Base Batch Normalization --- #
def bn_inference_cython(x: np.ndarray, 
                        running_mean: np.ndarray, 
                        inv_std: np.ndarray, 
                        gamma: np.ndarray, 
                        beta: np.ndarray) -> np.ndarray:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta

    y: np.ndarray = np.zeros_like(x, order="C", dtype=x.dtype)    

    try:
        bn_inference_cython_inner(y, x, running_mean, inv_std, gamma, beta)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_inference_cython\". Error: {e}")    

    return y 
# --- END bn_inference_cython --- #

def bn_inference_cython_inner(np.ndarray[npDT, ndim=2] y, 
                              np.ndarray[npDT, ndim=2] x, 
                              np.ndarray[npDT, ndim=1] running_mean, 
                              np.ndarray[npDT, ndim=1] inv_std, 
                              np.ndarray[npDT, ndim=1] gamma, 
                              np.ndarray[npDT, ndim=1] beta):
    cdef npDT[:,:] y_view = y
    cdef const npDT[:,:] x_view = x
    cdef const npDT[:] running_mean_view = running_mean
    cdef const npDT[:] inv_std_view = inv_std
    cdef const npDT[:] gamma_view = gamma
    cdef const npDT[:] beta_view = beta

    try:
        _bn_inference_cython_inner(y_view, x_view, running_mean_view, inv_std_view, gamma_view, beta_view)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_inference_cython\". Error: {e}")    

    return y 
# --- END bn_inference_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _bn_inference_cython_inner(npDT[:,:] y,
                                const npDT[:,:] x,
                                const npDT[:] running_mean,
                                const npDT[:] inv_std,                               
                                const npDT[:] gamma,
                                const npDT[:] beta):
    cdef int i, j = 0
    cdef npDT tmp
    
    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = (tmp * gamma[j]) + beta[j]
# --- END _bn_inference_cython_inner --- #

# --- END Base Batch Normalization --- #



# ==================================== #


# ==================================== #



# --- NCHW Batch Normalization --- #
def bn_inference_nchw_cython(x: np.ndarray, 
                             running_mean: np.ndarray, 
                             inv_std: np.ndarray, 
                             gamma: np.ndarray, 
                             beta: np.ndarray) -> np.ndarray:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta

    y: np.ndarray = np.zeros_like(x, order="C", dtype=x.dtype)

    try:
        bn_inference_nchw_cython_inner(y, x, running_mean, inv_std, gamma, beta)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_inference_nchw_cython\". Error: {e}")  

    return y
# --- END bn_inference_nchw_cython --- #

def bn_inference_nchw_cython_inner(np.ndarray[npDT, ndim=4] y,
                                   np.ndarray[npDT, ndim=4] x,
                                   np.ndarray[npDT, ndim=1] running_mean,
                                   np.ndarray[npDT, ndim=1] inv_std,
                                   np.ndarray[npDT, ndim=1] gamma,
                                   np.ndarray[npDT, ndim=1] beta):

    cdef npDT[:,:,:,:] y_view = y
    cdef const npDT[:,:,:,:] x_view = x
    cdef const npDT[:] running_mean_view = running_mean
    cdef const npDT[:] inv_std_view = inv_std
    cdef const npDT[:] gamma_view = gamma
    cdef const npDT[:] beta_view = beta

    _bn_inference_nchw_cython_inner(y_view, x_view, running_mean_view, inv_std_view, gamma_view, beta_view)
# --- END bn_inference_nchw_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _bn_inference_nchw_cython_inner(npDT[:,:,:,:] y,
                                     const npDT[:,:,:,:] x,
                                     const npDT[:] running_mean,
                                     const npDT[:] inv_std,
                                     const npDT[:] gamma,
                                     const npDT[:] beta):
    cdef int i, j, h, w
    cdef npDT tmp

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    tmp = (x[i, j, h, w] - running_mean[j]) * inv_std[j]
                    y[i, j, h, w] = (tmp * gamma[j]) + beta[j]
# --- END _bn_inference_nchw_cython_inner --- #

# --- END NCHW Batch Normalization --- #


# ==================================== #



# ==================================== #


# --- END ReLU Batch Normalization --- #

def bn_relu_inference_cython(x: np.ndarray, 
                             running_mean: np.ndarray, 
                             inv_std: np.ndarray, 
                             gamma: np.ndarray, 
                             beta: np.ndarray) -> np.ndarray:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta
    
    y:np.ndarray = np.zeros_like(x, order="C", dtype=x.dtype)

    try:
        bn_relu_inference_cython_inner(y, x, running_mean, inv_std, gamma, beta)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_relu_inference_cython\". Error: {e}")    

    return y
# --- END bn_relu_inference_cython --- #

def bn_relu_inference_cython_inner(np.ndarray[npDT, ndim=2] y,
                                   np.ndarray[npDT, ndim=2] x,
                                   np.ndarray[npDT, ndim=1] running_mean,
                                   np.ndarray[npDT, ndim=1] inv_std,                                   
                                   np.ndarray[npDT, ndim=1] gamma,
                                   np.ndarray[npDT, ndim=1] beta):
    
    cdef npDT[:,:] y_view = y
    cdef const npDT[:,:] x_view = x
    cdef const npDT[:] running_mean_view = running_mean
    cdef const npDT[:] inv_std_view = inv_std
    cdef const npDT[:] gamma_view = gamma
    cdef const npDT[:] beta_view = beta    

    _bn_relu_inference_cython_inner(y_view, x_view, running_mean_view, inv_std_view, gamma_view, beta_view)

# --- END bn_relu_inference_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _bn_relu_inference_cython_inner(npDT[:,:] y,
                                     const npDT[:,:] x,
                                     const npDT[:] running_mean,
                                     const npDT[:] inv_std,                                    
                                     const npDT[:] gamma,
                                     const npDT[:] beta):
    
    cdef int i, j = 0
    cdef npDT tmp

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = max((tmp * gamma[j]) + beta[j], 0)
# --- END _bn_relu_inference_cython_inner --- #

# --- END ReLU Batch Normalization --- #
