#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

def eltw_sum_cython(x_acc, x):
    shape = x_acc.shape

    if x.dtype == np.int8:
        eltw_sum_cython_inner_int8(x_acc.reshape(-1), x.reshape(-1))
    elif x.dtype == np.float32:
        eltw_sum_cython_inner_float32(x_acc.reshape(-1), x.reshape(-1))
    elif x.dtype == np.float64:
        eltw_sum_cython_inner_float64(x_acc.reshape(-1), x.reshape(-1))
    else:
        raise TypeError("Type '{}' is not supported by eltw_sum_cython!" % (str(x.dtype)))

    return x_acc.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eltw_sum_cython_inner_int8(np.ndarray[np.int8_t, ndim=1] x_acc,
                                np.ndarray[np.int8_t, ndim=1] x):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        x_acc[i] += x[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eltw_sum_cython_inner_float32(np.ndarray[np.float32_t, ndim=1] x_acc,
                                   np.ndarray[np.float32_t, ndim=1] x):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        x_acc[i] += x[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef eltw_sum_cython_inner_float64(np.ndarray[np.float64_t, ndim=1] x_acc,
                                   np.ndarray[np.float64_t, ndim=1] x):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        x_acc[i] += x[i]
