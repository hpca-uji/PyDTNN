import cython
import numpy as np

cimport numpy as np
from cython.parallel cimport prange

from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "compute_dense_acc_cython",
    "intersect_1d_indexes_cython",
    "reset_residuals_cython",
    "update_dense_weights_cython",
    "update_sparsed_weights_cython",
    "update_sparsed_weights_mv_cython"
)


def compute_dense_acc_cython(npDT[:,::1] residuals,
                             npDT[:,::1] dw,
                             npDT[:,::1] acc,
                             float learning_rate):

    cdef int i, j
    # cdef np.ndarray[np.float32_t, ndim=2] acc = np.empty_like(dw)

    for i in prange(dw.shape[0], nogil=True):
        for j in range(dw.shape[1]):
            acc[i, j] = residuals[i, j] + (<npDT> (learning_rate * dw[i, j]))


def intersect_1d_indexes_cython(np.int32_t[::1] local_indexes,
                                np.int32_t[::1] global_indexes,
                                np.int32_t[::1] intersected_indexes):

    cdef int count = 0
    cdef int i_local = 0
    cdef int i_global = 0

    while i_local < len(local_indexes) and i_global < len(global_indexes):
        if local_indexes[i_local] == global_indexes[i_global]:
            intersected_indexes[count] = global_indexes[i_global]
            i_local += 1
            i_global += 1
            count += 1
        elif local_indexes[i_local] > global_indexes[i_global]:
            i_global += 1
        else:  # if local_indexes[i_local] < global_indexes[i_global]:
            i_local += 1
    return intersected_indexes[:count]


def reset_residuals_cython(npDT[:,::1] acc,
                           np.int32_t[::1] indexes):
    cdef int i

    for i in prange(indexes.shape[0], nogil=True):
        acc[indexes[i]] = (<npDT> 0.0)


def update_dense_weights_cython(npDT[:,::1] w,
                                int nprocs,
                                npDT[:,::1] u):

    cdef int i, j
    cdef int rows = w.shape[0]
    cdef int cols = w.shape[1]

    for i in prange(rows, nogil=True):
        for j in range(cols):
            w[i, j] -= u[i, j] / nprocs


def update_sparsed_weights_cython(npDT[::1] w,
                                  int nprocs,
                                  npDT[::1] grads_to_update,
                                  np.int32_t[::1] indexes_to_update):


    cdef int i
    for i in prange(grads_to_update.shape[0], nogil=True):
        w[indexes_to_update[i]] -= grads_to_update[i] / nprocs


def update_sparsed_weights_mv_cython(npDT[::1] w,
                                     int nprocs,
                                     npDT[::1] grads_to_update,
                                     np.int32_t[::1] indexes_to_update,
                                     npDT[::1] velocity,
                                     float momentum):

    cdef int i, j
    cdef int index

    for i in prange(velocity.shape[0], nogil=True):
            velocity[i] = <npDT> (velocity[i] * momentum)

    for i in prange(grads_to_update.shape[0], nogil=True):
        index = indexes_to_update[i]

        velocity[index] += grads_to_update[i] / nprocs
        w[index] -= velocity[index]
