import cython
import numpy as np

cimport numpy as np

from pydtnn.backends.cython.utils.base cimport npDT


def summ_coo_cython(npDT[::1] self_data,
                    np.int32_t[::1] self_indices,
                    npDT[::1] other_data,
                    np.int32_t[::1] other_indices,
                    npDT[::1] summ_data,
                    np.int32_t[::1] summ_indices) -> tuple[np.ndarray, np.ndarray]:

    cdef int i_self = 0
    cdef int i_other = 0
    cdef int i_summ = 0

    # Adding the coincidences of both matrices
    while i_self < len(self_data) and i_other < len(other_data):
        if self_indices[i_self] == other_indices[i_other]:
            summ_data[i_summ] = self_data[i_self] + other_data[i_other]
            other_indices[i_summ] = i_self
            i_self += 1
            i_other += 1
            if summ_data[i_summ] == 0:
                # Case: "self_data[i_self] = -other_data[i_other]" ==>
                #  0 must not be stored (it will be replaced in the next iteration or sliced at the end)
                i_summ -= 1
        elif self_indices[i_self] > other_indices[i_other]:
            summ_data[i_summ] = other_data[i_other]
            other_indices[i_summ] = i_other
            i_other += 1
        else:  # if self_indices[i_self] < other_indices[i_other]:
            summ_data[i_summ] = self_data[i_self]
            other_indices[i_summ] = i_self
            i_self += 1
        i_summ += 1

    # NOTE: self or other still have values to iterate, but the other one has no more.
    # Adding the leftovers of this (self) matrix
    while i_self < len(self_data):
        summ_data[i_summ] = self_data[i_self]
        other_indices[i_summ] = i_self
        i_self += 1
        i_summ += 1
    
    # Adding the leftovers of the other matrix
    while i_other < len(other_data):
        summ_data[i_summ] = other_data[i_other]
        other_indices[i_summ] = i_other
        i_other += 1
        i_summ += 1

    return summ_data[:i_summ], summ_indices[:i_summ]


def top_threshold_selection_dense_cython(npDT[:, ::1] matrix,
                                         npDT threshold,
                                         npDT[::1] top_values,
                                         np.int32_t[::1] top_indices) -> tuple[np.ndarray, np.ndarray]:
    
    cdef int rows = matrix.shape[0]
    cdef int cols = matrix.shape[1]
    cdef int i, j, count = 0

    # Storing the top values and their indexes
    count = 0
    for i in range(rows):
        for j in range(cols):
            #if abs(matrix[i, j]) > threshold: NOTE: abs doesn't works with int8 and without gil 
            if matrix[i, j] > threshold or matrix[i, j] < -threshold:
                top_values[count] = matrix[i, j]
                top_indices[count] = i * cols + j
                count += 1
    return top_values[:count], top_indices[:count]


def top_threshold_selection_coo_cython(npDT[::1] values,
                                       np.int32_t[::1] indices,
                                       npDT threshold,
                                       npDT[::1] top_values,
                                       np.int32_t[::1] top_indices) -> tuple[np.ndarray, np.ndarray]:
    cdef int i, count = 0
    cdef int len_values = len(values)
    # Storing the values above the threshold.
    for i in range(len_values):
        #if abs(values[i]) > threshold: NOTE: abs doesn't works with int8 and without gil
        if values[i] > threshold or values[i] < -threshold:
            top_values[count] = values[i]
            top_indices[count] = indices[i]
            count += 1
    return top_values[:count], top_indices[:count]
