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
    "eltw_sum_cython"
)

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
def eltw_sum_cython(np.ndarray[npDT, ndim=4] x_acc, 
                    np.ndarray[npDT, ndim=4] x) -> np.ndarray:

    cdef np.ndarray[npDT, ndim=1] x_acc_reshaped = x_acc.reshape(-1, copy=False)
    cdef np.ndarray[npDT, ndim=1] x_reshaped = x.reshape(-1, copy=False)

    cdef int i
    for i in prange(x.shape[0], nogil=True):
        x_acc_reshaped[i] += x_reshaped[i]

    return x_acc
# --- END eltw_sum_cython --- #
