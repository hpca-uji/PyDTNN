import cython
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from pydtnn.backends.cython.utils.base cimport npDT

def summ_coo_cython(npDT[::1] self_data,
                    np.int32_t[::1] self_indices,
                    npDT[::1] other_data,
                    np.int32_t[::1] other_indices):

    cdef int i_self = 0
    cdef int i_other = 0
    cdef int i_summ = 0
    cdef int max_size = len(self_data) + len(other_data)
    cdef np.ndarray[npDT, ndim=1] summ_val = np.zeros(max_size, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] summ_indices = np.zeros(max_size, dtype=np.int32)

    # Adding the coincidences of both matrices
    while i_self < len(self_data) and i_other < len(other_data):
        if self_indices[i_self] == other_indices[i_other]:
            summ_val[i_summ] = self_data[i_self] + other_data[i_other]
            other_indices[i_summ] = i_self
            i_self += 1
            i_other += 1
            if summ_val[i_summ] == 0:
                # Case: "self_data[i_self] = -other_data[i_other]" ==>
                #  0 must not be stored (it will be replaced in the next iteration or sliced at the end)
                i_summ -= 1
        elif self_indices[i_self] > other_indices[i_other]:
            summ_val[i_summ] = other_data[i_other]
            other_indices[i_summ] = i_other
            i_other += 1
        else:  # if self_indices[i_self] < other_indices[i_other]:
            summ_val[i_summ] = self_data[i_self]
            other_indices[i_summ] = i_self
            i_self += 1
        i_summ += 1

    # NOTE: self or other still have values to iterate, but the other one has no more.
    # Adding the leftovers of this (self) matrix
    while i_self < len(self_data):
        summ_val[i_summ] = self_data[i_self]
        other_indices[i_summ] = i_self
        i_self += 1
        i_summ += 1
    
    # Adding the leftovers of the other matrix
    while i_other < len(other_data):
        summ_val[i_summ] = other_data[i_other]
        other_indices[i_summ] = i_other
        i_other += 1
        i_summ += 1

    return summ_val[:i_summ], summ_indices[:i_summ]


def top_threshold_selection_dense_cython(npDT[:, ::1] matrix,
                                         npDT threshold):
    
    cdef int rows = matrix.shape[0]
    cdef int cols = matrix.shape[1]
    cdef int i, j, count = 0
    cdef np.ndarray[np.int32_t, ndim=1]  count_vector = np.zeros(rows + 1, dtype=np.int32)

    # Counting the number of elements above the threshold
    for i in prange(rows, nogil=True):
        for j in range(cols):
            #if abs(matrix[i, j]) > threshold: NOTE: abs doesn't work in int8
            if matrix[i, j] > threshold or matrix[i, j] < -threshold:
                count_vector[i] += 1

    # Accumulating the count of elements above the threshold
    for i in range(rows):
        count_vector[rows] += count_vector[i]
    count = count_vector[rows]

    cdef np.ndarray[npDT, ndim=1] top_values = np.zeros(count, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] top_indices = np.zeros(count, dtype=np.int32)
    
    # Storing the top values and their indexes
    count = 0
    for i in range(rows):
        for j in range(cols):
            #if abs(matrix[i, j]) > threshold: NOTE: abs doesn't work in int8
            if matrix[i, j] > threshold or matrix[i, j] < -threshold:
                top_values[count] = matrix[i, j]
                top_indices[count] = i * cols + j
                count = 1
    return top_values, top_indices


def top_threshold_selection_coo_cython(np.ndarray[npDT, ndim=1] values,
                                       np.ndarray[np.int32_t, ndim=1] indices,
                                       npDT threshold):
    cdef int i, count = 0
    cdef int len_values = len(values)

    # Calculating the number of elements above the threshold.
    for i in prange(len_values, nogil=True):
        #if abs(values[i]) > threshold: NOTE: abs doesn't work in int8
        if values[i] > threshold or values[i] < -threshold:
            count += 1

    cdef np.ndarray[npDT, ndim=1] top_values = np.zeros(count, dtype=np.float32)
    cdef np.ndarray[np.int32_t, ndim=1] top_indices = np.zeros(count, dtype=np.int32)

    # Storing the values above the threshold.
    count = 0
    for i in range(len_values):
        #if abs(values[i]) > threshold: NOTE: abs doesn't work in int8
        if values[i] > threshold or values[i] < -threshold:
            top_values[count] = values[i]
            top_indices[count] = indices[i]
            count += 1
    return top_values, top_indices
