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
# --- END COMMON --- #

# ================== #
# ====== NHWC ====== #
# ================== #

def add_nhwc_cython(x: np.ndarray, 
                    b: np.ndarray) -> np.ndarray:
    try:
        add_nhwc(x, b)
        return x
    except TypeError as e:
        raise TypeError(f"Function: \"add_nhwc_cython\". Error: {e}")
# --- END add_nhwc_cython --- #

def add_nhwc(np.ndarray[npDT, ndim=2] x, 
             np.ndarray[npDT, ndim=1] b):
    cdef npDT[:,:] x_view = x
    cdef const npDT[:] b_view = b

    _add_nhwc(x_view, b_view)
# --- END add_nhwc --- #


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _add_nhwc(npDT[:,:] x,
               const npDT[:] b):

    cdef int i, j

    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[j]
# --- END _add_nhwc --- #

# ================== #
# ====== NCHW ====== #
# ================== #


def add_nchw_cython(x: np.ndarray, 
                    b: np.ndarray) -> np.ndarray:
  
    try:
        add_nchw(x, b)
        return x
    except TypeError as e:
        raise TypeError(f"Function: \"add_nchw_cython\". Error: {e}")
# --- END add_nchw_cython --- #

def add_nchw(np.ndarray[npDT, ndim=2] x, 
             np.ndarray[npDT, ndim=1] b):

    cdef npDT[:,:] x_view = x
    cdef const npDT[:] b_view = b

    _add_nchw(x_view, b_view)
# --- END _adadd_nchwd_nchw --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _add_nchw(npDT[:,:] x,
               const npDT[:] b):

    cdef int i, j
    
    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[i]
# --- END _add_nchw --- #
