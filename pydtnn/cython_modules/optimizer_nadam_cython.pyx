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
cimport openmp

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #

# NOTE: The implementation using Numpy performs better than this one with the same memory use.
def nadam_cython(w: np.ndarray, dw: np.ndarray, m: np.ndarray, v: np.ndarray,
                 float lr, float beta1, float beta2, float epsilon, float decay, 
                 int it) -> None:
    # NOTE: This function will modify the input's arrays.

    # w.shape === dw.shape === m.shape === v.shape
    # NOTE: x.shape is considered "npy_intp*" and it's not possible to do a direct cast to python's list nor tuple.
    shape = [w.shape[i] for i in range(w.ndim)]
    num_elemts = np.prod(shape)
    try:
        nadam(w.reshape(-1, copy=False), dw.reshape(-1, copy=False), 
              m.reshape(-1, copy=False), v.reshape(-1, copy=False), 
              lr, beta1, beta2, epsilon, decay, it, num_elemts)
    except TypeError as e:
        raise TypeError(f"Function: \"nadam_cython\". Error: {e}")
# --- END nadam_cython --- #

def nadam(np.ndarray[npDT, ndim=1] w, np.ndarray[npDT, ndim=1] dw, 
          np.ndarray[npDT, ndim=1] m, np.ndarray[npDT, ndim=1] v,
          float lr, float beta1, float beta2, float epsilon, 
          float decay, int it, int num_elemts):
    
    cdef npDT[:] w_view = w
    cdef npDT[:] dw_view = dw
    cdef npDT[:] m_view = m
    cdef npDT[:] v_view = v

    _nadam(w_view, dw_view, m_view, v_view, lr, beta1, beta2, epsilon, decay, it, num_elemts)
# --- END nadam --- #


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision
cdef _nadam(npDT[:] w, npDT[:] dw, npDT[:] m, npDT[:] v, 
            float lr, float beta1, float beta2, float epsilon, 
            float decay, int it, int num_elemts):

    cdef int i
    cdef npDT mt, vt
    cdef int num_threads


    for i in prange(num_elemts, nogil=True):
        m[i] = <npDT> (beta1 * m[i] + (1 - beta1) * dw[i])
        v[i] = <npDT> (beta2 * v[i] + (1 - beta2) * (dw[i] ** 2))
        
        mt = <npDT> ((m[i] + (1 - beta1) * dw[i]) / (1 - beta1**it))
        vt = <npDT> (v[i] / (1 - beta2**it))

        w[i] -= <npDT> (lr * (decay * w[i] + (mt / sqrt(vt + epsilon)) ))
# --- END _nadam --- #

