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

def argmax_cython(x: np.ndarray, int axis=0) -> tuple(np.ndarray, tuple(np.ndarray, np.ndarray)):
    
    if axis == 0: x = x.T

    maxv: np.ndarray = np.empty((x.shape[0],), dtype=x.dtype)
    amax: np.ndarray = np.empty((x.shape[0],), dtype=np.int32)
    rng: np.ndarray = np.empty((x.shape[0],), dtype=np.int32)    
    
    try:
        argmax_cython_inner(x, maxv, amax, rng)
        return maxv, tuple([amax, rng] if axis == 0 else [rng, amax])

    except TypeError as e:
        raise TypeError(f"Function: \"argmax_cython\". Error: {e}")    
# --- END argmax_cython --- #

def argmax_cython_inner(np.ndarray[npDT, ndim=2] x,
                        np.ndarray[npDT, ndim=1] maxv,
                        np.ndarray[np.int32_t, ndim=1] amax,
                        np.ndarray[np.int32_t, ndim=1] rng):

    cdef const npDT[:,:] x_view = x
    cdef npDT[:] maxv_view = maxv

    cdef npDT minval = np.iinfo(x.dtype).min if np.issubdtype(x.dtype, np.integer) else np.finfo(x.dtype).min    

    _argmax_cython_inner(x_view, maxv_view, amax, rng, minval)
# --- END argmax_cython_inner --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _argmax_cython_inner(const npDT[:,:] x,
                          npDT[:] maxv,
                          np.ndarray[np.int32_t, ndim=1] amax,
                          np.ndarray[np.int32_t, ndim=1] rng, 
                          npDT minval):

    cdef int i, j, idx_maxval
    cdef npDT maxval

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i, j] > maxval:
                maxval, idx_maxval = x[i, j], j
        amax[i], maxv[i], rng[i] = idx_maxval, maxval, i
# --- END _argmax_cython_inner --- #
