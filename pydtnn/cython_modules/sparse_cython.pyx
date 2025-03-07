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
cimport numpy as np
from cython.parallel cimport prange



@cython.boundscheck(False)
@cython.wraparound(False)
def summ_coo_cython(np.ndarray [np.float32_t, ndim=1] self_data,
                    np.ndarray [np.int32_t, ndim=1] self_rows,
                    np.ndarray [np.int32_t, ndim=1] self_cols,
                    np.ndarray [np.float32_t, ndim=1] other_data,
                    np.ndarray [np.int32_t, ndim=1] other_rows,
                    np.ndarray [np.int32_t, ndim=1] other_cols):

    cdef int count = 0
    cdef int i_self = 0
    cdef int i_other = 0
    cdef int row_self, row_other, col_self, col_other
    cdef int max_size = len(self_data) + len(other_data)
    cdef np.ndarray[np.float32_t, ndim=1] summ_val = np.empty(max_size, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] summ_row = np.empty(max_size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] summ_col = np.empty(max_size, dtype=np.int32)

    while i_self < len(self_data) or i_other < len(other_data):
        if i_other >= len(other_data):
            # There are only elements left in self
            summ_val[count] = self_data[i_self]
            summ_row[count] = self_rows[i_self]
            summ_col[count] = self_cols[i_self]
            i_self += 1
        elif i_self >= len(self_data):
            # There are only elements left in other
            summ_val[count] = other_data[i_other]
            summ_row[count] = other_rows[i_other]
            summ_col[count] = other_cols[i_other]
            i_other += 1
        else:
            row_self = self_rows[i_self]
            row_other = other_rows[i_other]
            if row_self < row_other:
                # Set self_data, self_row, self_col
                summ_val[count] = self_data[i_self]
                summ_row[count] = self_rows[i_self]
                summ_col[count] = self_cols[i_self]
                i_self += 1
            elif row_self > row_other:
                # Set other_data, other_row, other_col
                summ_val[count] = other_data[i_other]
                summ_row[count] = other_rows[i_other]
                summ_col[count] = other_cols[i_other]
                i_other += 1
            else:
                # Same row, let's see the column
                col_self = self_cols[i_self]
                col_other = other_cols[i_other]
                if col_self < col_other:
                    # Set self_data, self_row, self_col
                    summ_val[count] = self_data[i_self]
                    summ_row[count] = self_rows[i_self]
                    summ_col[count] = self_cols[i_self]
                    i_self += 1
                elif col_self > col_other:
                    # Set other_data, other_row, other_col
                    summ_val[count] = other_data[i_other]
                    summ_row[count] = other_rows[i_other]
                    summ_col[count] = other_cols[i_other]
                    i_other += 1
                else:
                    # Set self + other data, any row, any col
                    summ_val[count] = self_data[i_self] + other_data[i_other]
                    summ_row[count] = self_rows[i_self]
                    summ_col[count] = self_cols[i_self]
                    i_other += 1                    
                    i_self += 1
            count += 1

    return summ_val[:count], summ_row[:count], summ_col[:count]

    


@cython.boundscheck(False)
@cython.wraparound(False)
def top_threshold_selection_dense_cython(np.ndarray[np.float32_t, ndim=2] matrix, 
                                         float threshold):
    
    cdef int rows = matrix.shape[0]
    cdef int cols = matrix.shape[1]
    cdef int i, j, count = 0
    cdef np.ndarray[np.int32_t, ndim=1]  count_vector = np.zeros(rows + 1, dtype=np.int32)

    for i in prange(rows, nogil=True):
        for j in range(cols):
            if abs(matrix[i, j]) >= threshold:
                count_vector[i + 1] += 1

    for i in range(rows):
        count_vector[i + 1] += count_vector[i] 
    count = count_vector[rows]

    cdef np.ndarray[np.float32_t, ndim=1] top_values = np.empty(count, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] row_indices = np.empty(count, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col_indices = np.empty(count, dtype=np.int32)

    for i in prange(rows, nogil=True):
        for j in range(cols):
            if abs(matrix[i, j]) >= threshold:
                top_values[count_vector[i]] = matrix[i, j]
                row_indices[count_vector[i]] = i
                col_indices[count_vector[i]] = j
                count_vector[i] += 1

    return top_values, row_indices, col_indices


@cython.boundscheck(False)
@cython.wraparound(False)
def top_threshold_selection_coo_cython(np.ndarray[np.float32_t, ndim=1] values, 
                                       np.ndarray[np.int32_t, ndim=1] rows, 
                                       np.ndarray[np.int32_t, ndim=1] cols, 
                                       float threshold):
    cdef int i, count = 0
    cdef int len_values = len(values)

    for i in prange(len_values, nogil=True):
        if abs(values[i]) >= threshold:
            count += 1

    cdef np.ndarray[np.float32_t, ndim=1] top_values = np.empty(count, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] row_indices = np.empty(count, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col_indices = np.empty(count, dtype=np.int32)

    count = 0
    for i in range(len_values):
        if abs(values[i]) >= threshold:
            top_values[count] = values[i]
            row_indices[count] = rows[i]
            col_indices[count] = cols[i]
            count += 1
            
    return top_values, row_indices, col_indices

