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


import cython
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange


@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_2d_indexes_cython(cnp.ndarray[cnp.int32_t, ndim=1] local_rows,
                                cnp.ndarray[cnp.int32_t, ndim=1] local_cols,
                                cnp.ndarray[cnp.int32_t, ndim=1] global_rows,
                                cnp.ndarray[cnp.int32_t, ndim=1] global_cols):
    
    cdef int i, j
    cdef int local_size = local_rows.shape[0]
    cdef int global_size = global_rows.shape[0]
    cdef int count = 0

    for i in prange(local_size, nogil=True):
        for j in range(global_size):
            if local_rows[i] == global_rows[j] and local_cols[i] == global_cols[j]:
                count += 1

    cdef cnp.ndarray[cnp.int32_t, ndim=1] intersection_rows = np.empty(count, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] intersection_cols = np.empty(count, dtype=np.int32)

    cdef int idx = 0
    for i in range(local_size):
        for j in range(global_size):
            if local_rows[i] == global_rows[j] and local_cols[i] == global_cols[j]:
                intersection_rows[idx] = local_rows[i]
                intersection_cols[idx] = local_cols[i]
                idx += 1

    return intersection_rows, intersection_cols
