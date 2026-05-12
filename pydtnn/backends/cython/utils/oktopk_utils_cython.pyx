import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport prange

from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "compute_dense_acc_cython",
    "intersect_2d_indexes_cython",
    "reset_residuals_cython",
    "update_dense_weights_cython",
    "update_sparsed_weights_cython",
    "update_sparsed_weights_mv_cython"
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def compute_dense_acc_cython(npDT[:,::1] residuals,
                             npDT[:,::1] dw,
                             npDT[:,::1] acc,
                             float learning_rate):

    cdef int i, j
    # cdef np.ndarray[np.float32_t, ndim=2] acc = np.empty_like(dw)

    for i in prange(dw.shape[0], nogil=True):
        for j in range(dw.shape[1]):
            acc[i, j] = residuals[i, j] + (<npDT> (learning_rate * dw[i, j]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def intersect_2d_indexes_cython(np.int32_t[::1] local_rows,
                                np.int32_t[::1] local_cols,
                                np.int32_t[::1] global_rows,
                                np.int32_t[::1] global_cols,
                                np.int32_t[::1] intersected_rows,
                                np.int32_t[::1] intersected_cols):

    cdef int count = 0
    cdef int i_local = 0
    cdef int i_global = 0

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
@cython.initializedcheck(False)
def reset_residuals_cython(npDT[:,::1] acc,
                           np.int32_t[::1] rows,
                           np.int32_t[::1] cols):
    cdef int i

    for i in prange(rows.shape[0], nogil=True):
        acc[rows[i], cols[i]] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def update_dense_weights_cython(npDT[:,::1] w,
                                npDT[:,::1] u):

    cdef int i, j
    cdef int rows = w.shape[0]
    cdef int cols = w.shape[1]

    for i in prange(rows, nogil=True):
        for j in range(cols):
            w[i, j] -= u[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def update_sparsed_weights_cython(npDT[:,::1] w,
                                  npDT[::1] grads_to_update,
                                  np.int32_t[::1] rows_to_update,
                                  np.int32_t[::1] cols_to_update):


    cdef int i
    cdef int wi, wj

    for i in prange(grads_to_update.shape[0], nogil=True):
        wi = rows_to_update[i]
        wj = cols_to_update[i]
        w[wi, wj] -= grads_to_update[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def update_sparsed_weights_mv_cython(npDT[:,::1] w,
                                     npDT[::1] grads_to_update,
                                     np.int32_t[::1] rows_to_update,
                                     np.int32_t[::1] cols_to_update,
                                     npDT[:,::1] velocity,
                                     float momentum):

    cdef int i, j
    cdef int row_index, col_index

    for i in prange(velocity.shape[0], nogil=True):
        for j in range(velocity.shape[1]):
            velocity[i, j] = <npDT> (velocity[i, j] * momentum)

    for i in prange(grads_to_update.shape[0], nogil=True):
        row_index = rows_to_update[i]
        col_index = cols_to_update[i]

        velocity[row_index, col_index] += grads_to_update[i]
        w[row_index, col_index] -= velocity[row_index, col_index]
