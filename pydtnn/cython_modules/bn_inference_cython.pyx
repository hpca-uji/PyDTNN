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

__all__ = (
    "bn_inference_cython",
    "bn_inference_nchw_cython",
    "bn_relu_inference_cython"
)

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #

# --- Base Batch Normalization --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_inference_cython(npDT[:, ::1] x,
                        npDT[:, ::1] y,
                        npDT[::1] running_mean, 
                        npDT[::1] inv_std, 
                        npDT[::1] gamma, 
                        npDT[::1] beta) -> None:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta    

    cdef int i, j = 0
    cdef npDT tmp
    
    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = (tmp * gamma[j]) + beta[j]
# --- END bn_inference_cython --- #

# ==================================== #


# ==================================== #

# --- NCHW Batch Normalization --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_inference_nchw_cython(npDT[:, :, :, ::1] x, 
                             npDT[:, :, :, ::1] y,
                             npDT[::1] running_mean, 
                             npDT[::1] inv_std, 
                             npDT[::1] gamma, 
                             npDT[::1] beta) -> None:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta

    cdef int i, j, h, w
    cdef npDT tmp

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    tmp = (x[i, j, h, w] - running_mean[j]) * inv_std[j]
                    y[i, j, h, w] = (tmp * gamma[j]) + beta[j]

# --- END bn_inference_nchw_cython --- #

# ==================================== #



# ==================================== #


# --- END ReLU Batch Normalization --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_relu_inference_cython(npDT[:, ::1] x,
                             npDT[:, ::1] y,
                             npDT[::1] running_mean,
                             npDT[::1] inv_std,
                             npDT[::1] gamma,
                             npDT[::1] beta) -> None:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta
    
    # cdef np.ndarray[npDT, ndim=2] y = np.zeros_like(x, order="C", dtype=x.dtype)

    
    cdef int i, j = 0
    cdef npDT tmp

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = max((tmp * gamma[j]) + beta[j], 0)
# --- END bn_relu_inference_cython --- #

# --- END ReLU Batch Normalization --- #
