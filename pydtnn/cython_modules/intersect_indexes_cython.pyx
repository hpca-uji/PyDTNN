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


@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_2d_indexes_cython(cnp.ndarray[cnp.int32_t, ndim=1] local_rows,
                                cnp.ndarray[cnp.int32_t, ndim=1] local_cols,
                                cnp.ndarray[cnp.int32_t, ndim=1] global_rows,
                                cnp.ndarray[cnp.int32_t, ndim=1] global_cols):
    
    cdef int i, j
    cdef int local_size = local_rows.shape[0]
    cdef int global_size = global_rows.shape[0]

    cdef list intersection_rows = []
    cdef list intersection_cols = []

    for i in range(local_size):
        for j in range(global_size):
            if local_rows[i] == global_rows[j] and local_cols[i] == global_cols[j]:
                intersection_rows.append(local_rows[i])
                intersection_cols.append(local_cols[i])

    return np.array(intersection_rows, dtype=np.int32), np.array(intersection_cols, dtype=np.int32)
