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
ctypedef fused supported_types_t:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END supported_types_t -- #

# --- Base Batch Normalization --- #
def bn_inference_cython(np.ndarray[supported_types_t, ndim=2] x, 
                        np.ndarray[supported_types_t, ndim=1] running_mean, 
                        np.ndarray[supported_types_t, ndim=1] inv_std, 
                        np.ndarray[supported_types_t, ndim=1] gamma, 
                        np.ndarray[supported_types_t, ndim=1] beta) -> np.ndarray:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta

    cdef np.ndarray[supported_types_t, ndim=2] y = np.empty(x, order="C", dtype=x.dtype)

    cdef const supported_types_t[:,:] view_x = x
    cdef const supported_types_t[:] view_running_mean = running_mean
    cdef const supported_types_t[:] view_inv_std = inv_std
    cdef supported_types_t[:,:] view_y = y
    cdef const supported_types_t[:] view_gamma = gamma
    cdef const supported_types_t[:] view_beta = beta 

    try:
        bn_inference_cython_inner(view_x, view_running_mean, view_inv_std, view_y, view_gamma, view_beta)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_inference_cython\". Error: {e}")    

    return y 
# --- END bn_inference_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_inference_cython_inner(const supported_types_t[:,:] x,
                               const supported_types_t[:] running_mean,
                               const supported_types_t[:] inv_std,
                               supported_types_t[:,:] y,
                               const supported_types_t[:] gamma,
                               const supported_types_t[:] beta):
    cdef int i, j = 0
    cdef supported_types_t tmp
    
    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = (tmp * gamma[j]) + beta[j]
# --- END _bn_inference_cython_inner --- #

# --- END Base Batch Normalization --- #

# ==================================== #
# ==================================== #

# --- NCHW Batch Normalization --- #
def bn_inference_nchw_cython(np.ndarray[supported_types_t, ndim=4] x, 
                             np.ndarray[supported_types_t, ndim=1] running_mean, 
                             np.ndarray[supported_types_t, ndim=1] inv_std, 
                             np.ndarray[supported_types_t, ndim=1] gamma, 
                             np.ndarray[supported_types_t, ndim=1] beta) -> np.ndarray:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta

    cdef np.ndarray[supported_types_t, ndim=4] y = np.empty(x, order="C", dtype=x.dtype)

    cdef const supported_types_t[:,:,:,:] view_x = x
    cdef const supported_types_t[:] view_running_mean = running_mean
    cdef const supported_types_t[:] view_inv_std = inv_std
    cdef supported_types_t[:,:,:,:] view_y = y
    cdef const supported_types_t[:] view_gamma = gamma
    cdef const supported_types_t[:] view_beta = beta 

    try:
        bn_inference_nchw_cython_inner(view_x, view_running_mean, 
                                       view_inv_std, view_y, 
                                       view_gamma, view_beta)
    except TypeError as e:
        raise TypeError(f"Function: \"bn_inference_nchw_cython\". Error: {e}")  

    return y
# --- END bn_inference_nchw_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bn_inference_nchw_cython_inner(const supported_types_t[:,:,:,:] x,
                                    const supported_types_t[:] running_mean,
                                    const supported_types_t[:] inv_std,
                                    supported_types_t[:,:,:,:] y,
                                    const supported_types_t[:] gamma,
                                    const supported_types_t[:] beta):
    cdef int i, j, h, w
    cdef supported_types_t tmp

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    tmp = (x[i, j, h, w] - running_mean[j]) * inv_std[j]
                    y[i, j, h, w] = (tmp * gamma[j]) + beta[j]
# --- END bn_inference_nchw_cython_inner --- #

# --- END NCHW Batch Normalization --- #