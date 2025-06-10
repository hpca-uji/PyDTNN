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

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #

# NOTE: The implementation using Numpy performs better than this one with the same memory use.
def sgd_cython(w: np.ndarray, dw: np.ndarray, velocity: np.ndarray, 
               float lr, float momentum, nesterov:bool, float decay) -> None:
    # NOTE: This function will modify the input's arrays.

    # w.shape === dw.shape === m.shape === v.shape
    # NOTE: x.shape is considered "npy_intp*" and it's not possible to do a direct cast to python's list nor tuple.
    shape = [w.shape[i] for i in range(w.ndim)]
    num_elemts = np.prod(shape)
    try:
        sgd(w.reshape(-1, copy=False), dw.reshape(-1, copy=False), 
                velocity.reshape(-1, copy=False), lr, momentum, nesterov, decay, num_elemts)
    except TypeError as e:
        raise TypeError(f"Function: \"sgd_cython\". Error: {e}")
# --- END sgd_cython --- #

def sgd(np.ndarray[npDT, ndim=1] w, np.ndarray[npDT, ndim=1] dw, 
        np.ndarray[npDT, ndim=1] velocity, float lr, float momentum, 
        nesterov:bool, float decay, int num_elemts):
    
    cdef npDT[:] w_view = w
    cdef npDT[:] dw_view = dw
    cdef npDT[:] velocity_view = velocity

    if nesterov:
        _sgd_nesterov(w_view, dw_view, velocity_view, lr, momentum, decay, num_elemts)
    else:
        _sgd(w_view, dw_view, velocity_view, lr, momentum, decay, num_elemts)
# --- END sgd --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _sgd(npDT[:] w, npDT[:] dw, npDT[:] velocity,
              float lr, float momentum, 
              float decay, int num_elemts):

    cdef int i

    for i in prange(num_elemts, nogil=True):
        velocity[i] = <npDT> (momentum * velocity[i] + dw[i])
        w[i] -=  <npDT> (lr * (decay * w[i] + velocity[i]))
    # --- END _sgd --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _sgd_nesterov(npDT[:] w, npDT[:] dw, npDT[:] velocity,
              float lr, float momentum, 
              float decay, int num_elemts):

    cdef int i

    for i in prange(num_elemts, nogil=True):
        velocity[i] = <npDT> (momentum * velocity[i] + dw[i])
        w[i] -= <npDT> (lr * (decay * w[i] + dw[i] + momentum * velocity[i]))
# --- END _sgd_nesterov --- #
