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
def compute_dense_acc_cython(np.ndarray[np.float32_t, ndim=2] residuals, 
                             np.ndarray[np.float32_t, ndim=2] dw, 
                             float learning_rate):

    cdef int i, j

    for i in prange(dw.shape[0], nogil=True):
        for j in range(dw.shape[1]):
            dw[i, j] = residuals[i, j] + (learning_rate * dw[i, j])
    
    return dw


@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_2d_indexes_cython(np.ndarray [np.int32_t, ndim=1] local_rows,
                                np.ndarray [np.int32_t, ndim=1] local_cols,
                                np.ndarray [np.int32_t, ndim=1] global_rows,
                                np.ndarray [np.int32_t, ndim=1] global_cols):
    
    cdef int count = 0
    cdef int i_local = 0
    cdef int i_global = 0
    cdef int max_size = min(len(local_rows), len(global_rows))
    cdef np.ndarray[np.int32_t, ndim=1] intersected_rows = np.empty(max_size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] intersected_cols = np.empty(max_size, dtype=np.int32)

    while i_local < len(local_rows) and i_global < len(global_rows):
        local_row = local_rows[i_local]
        global_row = global_rows[i_global]
        if local_row < global_row:
            i_local += 1
        elif local_row > global_row:
            i_global += 1
        else:
            local_col = local_cols[i_local]
            global_col = global_cols[i_global]
            if local_col < global_col:
                i_local += 1
            elif local_col > global_col:
                i_global += 1
            else:
                intersected_rows[count] = local_row
                intersected_cols[count] = local_col
                i_global += 1                    
                i_local += 1
                count += 1
    return intersected_rows[:count], intersected_cols[:count]


@cython.boundscheck(False)
@cython.wraparound(False)
def reset_residuals_cython(np.ndarray[np.float32_t, ndim=2] acc, 
                           np.ndarray[np.int32_t, ndim=1] rows, 
                           np.ndarray[np.int32_t, ndim=1] cols):

    cdef int i

    for i in prange(rows.shape[0], nogil=True):
        acc[rows[i], cols[i]] = 0
    
    return acc


@cython.boundscheck(False)
@cython.wraparound(False)
def top_threshold_selection_cython(np.ndarray[np.float32_t, ndim=2] matrix, double threshold):
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

    return top_values, (row_indices, col_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def top_threshold_selection_coo_cython(np.ndarray[np.float32_t, ndim=1] values, 
                                       np.ndarray[np.int32_t, ndim=1] rows, 
                                       np.ndarray[np.int32_t, ndim=1] cols, 
                                       double threshold):
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
            
    return top_values, (row_indices, col_indices)


@cython.boundscheck(False)
@cython.wraparound(False)
def update_dense_weights_cython(np.ndarray[np.float32_t, ndim=2] w, 
                                np.ndarray[np.float32_t, ndim=2] u, 
                                int nprocs):

    cdef int i, j
    cdef int rows = w.shape[0]
    cdef int cols = w.shape[1]

    for i in prange(rows, nogil=True):
        for j in range(cols):
            w[i, j] = w[i, j] - (u[i, j] / nprocs)

    return w


@cython.boundscheck(False)
@cython.wraparound(False)
def update_sparsed_weights_cython(np.ndarray[np.float32_t, ndim=2] w, 
                                  np.ndarray[np.float32_t, ndim=1] grads_to_update, 
                                  np.ndarray[np.int32_t, ndim=1] rows_to_update, 
                                  np.ndarray[np.int32_t, ndim=1] cols_to_update, 
                                  int nprocs):


    cdef int idx, row, col
    cdef int num_updates = grads_to_update.shape[0]

    for idx in prange(num_updates, nogil=True):
        w[rows_to_update[idx], cols_to_update[idx]] -= grads_to_update[idx] / nprocs

    return w