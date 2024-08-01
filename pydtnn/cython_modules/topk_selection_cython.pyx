#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-24 Universitat Jaume I
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
cimport numpy as cnp
from libc.math cimport fabsf as fabs
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def topk_selection_cython(cnp.ndarray[cnp.float32_t, ndim=1] data, double threshold):
    cdef cnp.ndarray[cnp.float32_t, ndim=1] topk = np.zeros_like(data, dtype=np.float32)  
    cdef cnp.ndarray[cnp.int32_t, ndim=1] topk_indexes = np.empty(data.size, dtype=np.int32)  
    cdef int idx = 0
    cdef int total_elements = data.size
    cdef int i

    for i in range(total_elements):
        if fabs(data[i]) >= threshold:
            topk[i] = data[i]
            topk_indexes[idx] = i
            idx += 1

    topk_indexes = topk_indexes[:idx]
    return topk, topk_indexes



@cython.boundscheck(False)
@cython.wraparound(False)
def flattened_topk_selection_cython(cnp.ndarray[cnp.float32_t, ndim=1] data, double threshold):
    topk = np.zeros_like(data)
    topk_indexes = np.where(np.abs(data) >= threshold)[0]
    for idx in topk_indexes:
        topk[idx] = data[idx]
    return topk, topk_indexes

