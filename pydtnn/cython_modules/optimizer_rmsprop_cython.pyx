#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2025 Universitat Jaume I
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

# NOTE: The implementation using Numpy performs better than this one with the same memory use.
def rmsprop_cython(w: np.ndarray, dw: np.ndarray, cache: np.ndarray, 
                   float lr, float rho, float epsilon, float decay) -> None:
    # NOTE: This function will modify the input's arrays.

    # w.shape === dw.shape === m.shape === v.shape
    # NOTE: x.shape is considered "npy_intp*" and it's not possible to do a direct cast to python's list nor tuple.
    shape = [w.shape[i] for i in range(w.ndim)]
    num_elemts = np.prod(shape)
    try:
        rmsprop(w.reshape(-1, copy=False), dw.reshape(-1, copy=False), 
                cache.reshape(-1, copy=False), lr, rho, epsilon, decay, num_elemts)
    except TypeError as e:
        raise TypeError(f"Function: \"rmsprop_cython\". Error: {e}")
# --- END rmsprop_cython --- #

def rmsprop(np.ndarray[npDT, ndim=1] w, np.ndarray[npDT, ndim=1] dw, 
            np.ndarray[npDT, ndim=1] cache, float lr, float rho, 
            float epsilon, float decay, int num_elemts):
    
    cdef npDT[:] w_view = w
    cdef npDT[:] dw_view = dw
    cdef npDT[:] cache_view = cache

    _rmsprop(w_view, dw_view, cache_view, lr, rho, epsilon, decay, num_elemts)
# --- END rmsprop --- #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision
cdef _rmsprop(npDT[:] w, npDT[:] dw, npDT[:] cache,
              float lr, float rho, float epsilon, 
              float decay, int num_elemts):

    cdef int i

    for i in prange(num_elemts, nogil=True):
        cache[i] = <npDT> (rho * cache[i] + (1 - rho) * dw[i] ** 2)
        w[i] -= <npDT> (lr * (decay * w[i] + dw[i] / sqrt(cache[i] + epsilon)))
# --- END _rmsprop --- #

