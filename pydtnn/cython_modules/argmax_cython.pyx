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
    "argmax_cython"
)

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def argmax_cython(np.ndarray[npDT, ndim=2] x, 
                  npDT[:] maxv, 
                  np.int32_t[:] amax,
                  np.int32_t[:] rng,
                  int axis=0) -> tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
    
    if axis == 0: x = x.T

    cdef npDT minval = np.iinfo(x.dtype).min if np.issubdtype(x.dtype, np.integer) else np.finfo(x.dtype).min    

    cdef int i, j, idx_maxval
    cdef npDT maxval

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i, j] > maxval:
                maxval, idx_maxval = x[i, j], j
        amax[i], maxv[i], rng[i] = idx_maxval, maxval, i

    return (amax, rng) if axis == 0 else (rng, amax)
# --- END argmax_cython --- #
