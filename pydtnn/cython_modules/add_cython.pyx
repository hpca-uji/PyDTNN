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
# --- END COMMON --- #

# ============================= #
# ===== START NCHW FORMAT ===== #
# ============================= #

def add_nchw_cython(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    try:
        _add_nchw_cython_inner(x, b)
        return x
    except TypeError as e:
        raise TypeError(f"Function: \"adaptive_avg_pooling_fwd_nchw_cython\". Error: {e}")
# --- END add_nchw_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _add_nchw_cython_inner(np.ndarray[supported_types_t, ndim=2] x,
                            np.ndarray[supported_types_t, ndim=1] b):
    cdef supported_types_t[:, :] x_view = x
    cdef const supported_types_t[:] b_view = b
    _add_nchw_cython(x_view, b_view)
# --- END _add_nchw_cython_inner --- #
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _add_nchw_cython(supported_types_t[:,:] x,
                      const supported_types_t[:] b):
    cdef int i, j
    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[i]
# --- END _add_nchw_cython --- #

# ============================= #
# ====== END NCHW FORMAT ====== #
# ============================= #

# ============================= #

# ============================= #
# ===== START NHWC FORMAT ===== #
# ============================= #

def add_nhwc_cython(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    
    try:
        _add_nhwc_cython_inner(x, b)
        return x
    except TypeError as e:
        raise TypeError(f"Function: \"add_nhwc_cython\". Error: {e}")
# --- END add_nhwc_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _add_nhwc_cython_inner(np.ndarray[supported_types_t, ndim=2] x,
                            np.ndarray[supported_types_t, ndim=1] b):
    cdef supported_types_t[:, :] x_view = x
    cdef const supported_types_t[:] b_view = b
    _add_nhwc_cython(x_view, b_view)
# --- END _add_nhwc_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _add_nhwc_cython(supported_types_t[:,:] x,
                      const supported_types_t[:] b):
    cdef int i, j
    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[j]
# --- END _add_nhwc_cython --- #

# ============================= #
# ====== END NHWC FORMAT ====== #
# ============================= #

