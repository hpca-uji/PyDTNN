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
import indexing

# --- COMMON --- #
ctypedef fused supported_types_t:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END supported_types_t -- #

def argmax_cython(x:np.ndarray, axis=0) -> tuple(np.ndarray, tuple(np.ndarray, np.ndarray)):
    if axis == 0: x = x.T

    cdef np.ndarray maxv = np.empty((x.shape[0],), dtype=x.dtype)
    cdef np.ndarray amax = np.empty((x.shape[0],), dtype=np.int32)
    cdef np.ndarray rng = np.empty((x.shape[0],), dtype=np.int32)

    try:
        indexing.argmax_cython_inner[x.dtype](x, maxv, amax, rng)
        return maxv, tuple([amax, rng] if axis == 0 else [rng, amax])

    except TypeError as e:
        raise TypeError(f"Function: \"argmax_cython\". Error: {e}")    
# --- END argmax_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner(np.ndarray[supported_types_t, ndim=2] x,
                         np.ndarray[supported_types_t, ndim=1] maxv,
                         np.ndarray[np.int32_t, ndim=1] amax,
                         np.ndarray[np.int32_t, ndim=1] rng):

    cdef const supported_types_t[:,:] x_view = x
    cdef supported_types_t[:] maxv_view = maxv
    cdef np.int32_t[:] amax_view = amax
    cdef np.int32_t[:] rng_view = rng
    # TODO: Put this in the layer's initializer and pass it as a parameter to the 1st function
    cdef supported_types_t minval = np.iinfo(x.dtype).min if np.issubdtype(x.dtype, np.integer) else np.finfo(x.dtype).min

    _argmax_cython(x_view, maxv_view, amax_view, rng_view, minval)
# --- END argmax_cython_inner --- #


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _argmax_cython(const supported_types_t[:, :] x,
                    supported_types_t[:] maxv,
                    np.int32_t[:] amax,
                    np.int32_t[:] rng, 
                    supported_types_t minval):
    cdef int i, j, idx_maxval
    cdef supported_types_t maxval

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i, j] > maxval:
                maxval, idx_maxval = x[i, j], j
        amax[i], maxv[i], rng[i] = idx_maxval, maxval, i
# --- END _argmax_cython --- #
