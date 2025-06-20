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

def eltw_sum_cython(x_acc: np.ndarray, x: np.ndarray) -> np.ndarray:

    try:
        eltw_sum_cython_inner(x_acc.reshape(-1, copy=False), x.reshape(-1, copy=False))
    except TypeError as e:
        raise TypeError(f"Function: \"eltw_sum_cython\". Error: {e}")

    return x_acc
# --- END eltw_sum_cython --- #

def eltw_sum_cython_inner(np.ndarray[npDT, ndim=1] x_acc,
                          np.ndarray[npDT, ndim=1] x):

    cdef npDT[:] x_acc_view = x_acc
    cdef const npDT[:] x_view = x

    _eltw_sum_cython_inner(x_acc_view, x_view)
# --- END eltw_sum_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _eltw_sum_cython_inner(npDT[:] x_acc,
                            const npDT[:] x):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        x_acc[i] += x[i]
# --- END _eltw_sum_cython_inner --- #
