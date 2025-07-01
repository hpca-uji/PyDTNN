#
#  This file is part of Python Distributed Training of neural networks (PyDTnn)
#
#  copyright (c) 2021-2025 Universitat Jaume I
#
#  PyDTnn is free software: you can redistribute it and/or modify it under the
#  terms of the GnU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GnU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# =================== #
# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #
# =================== #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def pointwise_conv_cython(npDT[:,:,:,::1] x, npDT[:,::1] k) -> np.ndarray:

    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int co = k.shape[0]

    cdef npDT[:,:,:,::1] out = np.zeros((n, co, h, w))

    cdef int nn, cco, cc, ii, jj

    for cco in prange(co, nogil=True):
        for cc in range(c):
            for nn in range(n):
                for ii in range(h):
                    for jj in range(w):
                        out[nn, cco, ii, jj] += x[nn, cc, ii, jj] * k[cco, cc]

    return out
# --- END pointwise_conv_cython --- #
