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

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

###############################################
#                 relu_cython                 #
###############################################

def relu_cython(x: np.ndarray) -> tuple(np.ndarray, np.ndarray):
    
    # NOTE: x.shape is considered "npy_intp*" and it's not possible to do a direct cast to python's list nor tuple.
    shape = [x.shape[i] for i in range(x.ndim)]
    size = np.prod(shape)

    max: np.ndarray = np.zeros((size,), dtype=x.dtype)
    mask: np.ndarray = np.zeros((size,), dtype=np.int8)

    try:
        max, mask = relu_cython_template(x.reshape(-1), max, mask)        
        return max.reshape(shape), mask.reshape(shape)
    except KeyError as e:
        raise TypeError(f"Function: \"relu_cython\". Error: {e}")
# --- END relu_cython --- #

def relu_cython_template(np.ndarray[npDT, ndim=1] x, 
                         np.ndarray[npDT, ndim=1] max,
                         np.ndarray[np.int8_t, ndim=1] mask):
    cdef:                
        const npDT[:] x_view = x
        npDT[:] max_view = max
        np.int8_t[:] mask_view = mask
    relu_cython_inner(x_view, max_view, mask_view)
    return max, mask
# --- END relu_cython_template --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner(const npDT[:] x,
                       npDT[:] max,
                       np.int8_t[:] mask):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
# --- END relu_cython_inner --- # 


###############################################
#             capped_relu_cython              #
###############################################


# NOTE: If cap = 6, then this is a Relu6.
def capped_relu_cython(x: np.ndarray, cap: float) -> tuple(np.ndarray, np.ndarray):
    
    # NOTE: x.shape is considered "npy_intp*" and it's not possible to do a direct cast to python's list nor tuple.
    shape = [x.shape[i] for i in range(x.ndim)]
    size = np.prod(shape)

    max: np.ndarray = np.zeros((size,), dtype=x.dtype)
    mask: np.ndarray = np.zeros((size,), dtype=np.int8)

    try:
        max, mask = capped_relu_cython_template(x.reshape(-1), max, mask, cap)        
        return max.reshape(shape), mask.reshape(shape)
    except KeyError as e:
        raise TypeError(f"Function: \"capped_relu_cython\". Error: {e}")
# --- END capped_relu_cython --- #

def capped_relu_cython_template(np.ndarray[npDT, ndim=1] x, 
                         np.ndarray[npDT, ndim=1] max,
                         np.ndarray[np.int8_t, ndim=1] mask,
                         np.float64_t cap):
    cdef:                
        const npDT[:] x_view = x
        npDT[:] max_view = max
        np.int8_t[:] mask_view = mask
    capped_relu_cython_inner(x_view, max_view, mask_view, cap)
    return max, mask
# --- END capped_relu_cython_template --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef capped_relu_cython_inner(const npDT[:] x,
                       npDT[:] max,
                       np.int8_t[:] mask, 
                       np.float64_t cap):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] >= cap:
            max[i], mask[i] = <npDT> cap, 1
        elif x[i] > 0: # cap > x[i] > 0
            max[i], mask[i] = x[i], 1
# --- END capped_relu_cython_inner --- # 

###############################################
#              leaky_relu_cython              #
###############################################

def leaky_relu_cython(x: np.ndarray, negative_slope: float) -> tuple(np.ndarray, np.ndarray):
    
    # NOTE: x.shape is considered "npy_intp*" and it's not possible to do a direct cast to python's list nor tuple.
    shape = [x.shape[i] for i in range(x.ndim)]
    size = np.prod(shape)

    max: np.ndarray = np.zeros((size,), dtype=x.dtype)
    mask: np.ndarray = np.zeros((size,), dtype=np.float32)

    try:
        max, mask = leaky_relu_cython_template(x.reshape(-1), max, mask, negative_slope)        
        return max.reshape(shape), mask.reshape(shape)
    except KeyError as e:
        raise TypeError(f"Function: \"leaky_relu_cython\". Error: {e}")
# --- END leaky_relu_cython --- #

def leaky_relu_cython_template(np.ndarray[npDT, ndim=1] x, 
                         np.ndarray[npDT, ndim=1] max,
                         np.ndarray[np.float32_t, ndim=1] mask,
                         np.float32_t negative_slope):
    cdef:                
        const npDT[:] x_view = x
        npDT[:] max_view = max
        np.float32_t[:] mask_view = mask
    leaky_relu_cython_inner(x_view, max_view, mask_view, negative_slope)
    return max, mask
# --- END leaky_relu_cython_template --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef leaky_relu_cython_inner(const npDT[:] x,
                       npDT[:] max,
                       np.float32_t[:] mask, 
                       np.float32_t negative_slope):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
        elif x[i] < 0:
            max[i], mask[i] = <npDT> (x[i] * negative_slope), negative_slope
# --- END leaky_relu_cython_inner --- # 
