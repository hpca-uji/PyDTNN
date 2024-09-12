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

from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
cimport numpy as cnp
import numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def top_threshold_selection_cython(cnp.ndarray[cnp.float32_t, ndim=2] matrix, double threshold):

    cdef int rows = matrix.shape[0]
    cdef int cols = matrix.shape[1]
    cdef int i, j, count = 0

    for i in range(rows):
        for j in range(cols):
            if abs(matrix[i, j]) >= threshold:
                count += 1

    cdef cnp.ndarray[cnp.float32_t, ndim=1] top_values = np.empty(count, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] row_indices = np.empty(count, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] col_indices = np.empty(count, dtype=np.int32)

    count = 0
    for i in range(rows):
        for j in range(cols):
            if abs(matrix[i, j]) >= threshold:
                top_values[count] = matrix[i, j]
                row_indices[count] = i
                col_indices[count] = j
                count += 1

    return top_values, (row_indices, col_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def top_threshold_selection_coo_cython(cnp.ndarray[cnp.float32_t, ndim=1] values, 
                                       cnp.ndarray[cnp.int32_t, ndim=1] rows, 
                                       cnp.ndarray[cnp.int32_t, ndim=1] cols, 
                                       double threshold):

    cdef int i, count = 0

    for i in range(len(values)):
        if abs(values[i]) >= threshold:
            count += 1

    cdef cnp.ndarray[cnp.float32_t, ndim=1] top_values = np.empty(count, dtype=np.float32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] row_indices = np.empty(count, dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] col_indices = np.empty(count, dtype=np.int32)

    count = 0
    for i in range(len(values)):
        if abs(values[i]) >= threshold:
            top_values[count] = values[i]
            row_indices[count] = rows[i]
            col_indices[count] = cols[i]
            count += 1
            
    return top_values, (row_indices, col_indices)
