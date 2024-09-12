# cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.stdint cimport int32_t
import numpy as np
cimport numpy as cnp

def intersect_2d_indexes_cython(cnp.ndarray[int32_t, ndim=1] local_rows,
                                cnp.ndarray[int32_t, ndim=1] local_cols,
                                cnp.ndarray[int32_t, ndim=1] global_rows,
                                cnp.ndarray[int32_t, ndim=1] global_cols):
    
    # Definición de tipo para los arrays estructurados con cadenas para los tipos
    dtype = [('row', 'int32'), ('col', 'int32')]
    
    # Creación de arrays estructurados
    cdef cnp.ndarray local_indices = np.array(list(zip(local_rows, local_cols)), dtype=dtype)
    cdef cnp.ndarray global_indices = np.array(list(zip(global_rows, global_cols)), dtype=dtype)
    
    # Intersección de índices
    cdef cnp.ndarray intersection_indices = np.intersect1d(local_indices, global_indices, assume_unique=False)
    
    # Extraer filas y columnas
    intersection_rows = intersection_indices['row']
    intersection_cols = intersection_indices['col']

    return intersection_rows, intersection_cols