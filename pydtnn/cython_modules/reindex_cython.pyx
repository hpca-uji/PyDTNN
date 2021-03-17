#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def reindex_cython(v_new_indexes, h_new_indexes, matrix_in, matrix_out):
    """
    Implements a parallel version of:
        matrix_out = matrix_in[:, :, h_new_indexes, :]
        matrix_out = matrix_out[:, :, :, v_new_indexes]
    """
    if matrix_in.dtype == np.float32:
        if v_new_indexes is not None and h_new_indexes is not None:
            reindex_cython_float32(v_new_indexes, h_new_indexes, matrix_in, matrix_out)
        elif v_new_indexes is not None:
            reindex_only_rows_cython_float32(v_new_indexes, matrix_in, matrix_out)
        elif h_new_indexes is not None:
            reindex_only_columns_cython_float32(h_new_indexes, matrix_in, matrix_out)
        else:
            raise ValueError("reindex_cython() should not be called if there is nothing to reindex")
    else:
        raise TypeError("Type '{}' is not supported by reindex_cython".format(matrix_in.dtype))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef reindex_cython_float32(np.ndarray[np.int_t, ndim=1] v_new_indexes,
                            np.ndarray[np.int_t, ndim=1] h_new_indexes,
                            np.ndarray[np.float32_t, ndim=4] matrix_in,
                            np.ndarray[np.float32_t, ndim=4] matrix_out):
    cdef Py_ssize_t d0, d1, d2, d3
    for d0 in prange(matrix_out.shape[0], nogil=True, schedule="static"):
        for d1 in range(matrix_out.shape[1]):
            for d2 in range(matrix_out.shape[2]):
                for d3 in range(matrix_out.shape[3]):
                    matrix_out[d0, d1, d2, d3] = matrix_in[d0, d1, v_new_indexes[d2], h_new_indexes[d3]]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef reindex_only_rows_cython_float32(np.ndarray[np.int_t, ndim=1] v_new_indexes,
                                      np.ndarray[np.float32_t, ndim=4] matrix_in,
                                      np.ndarray[np.float32_t, ndim=4] matrix_out):
    cdef Py_ssize_t d0, d1, d2, d3
    for d0 in prange(matrix_out.shape[0], nogil=True, schedule="static"):
        for d1 in range(matrix_out.shape[1]):
            for d2 in range(matrix_out.shape[2]):
                for d3 in range(matrix_out.shape[3]):
                    matrix_out[d0, d1, d2, d3] = matrix_in[d0, d1, v_new_indexes[d2], d3]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef reindex_only_columns_cython_float32(np.ndarray[np.int_t, ndim=1] h_new_indexes,
                                         np.ndarray[np.float32_t, ndim=4] matrix_in,
                                         np.ndarray[np.float32_t, ndim=4] matrix_out):
    cdef Py_ssize_t d0, d1, d2, d3
    for d0 in prange(matrix_out.shape[0], nogil=True, schedule="static"):
        for d1 in range(matrix_out.shape[1]):
            for d2 in range(matrix_out.shape[2]):
                for d3 in range(matrix_out.shape[3]):
                    matrix_out[d0, d1, d2, d3] = matrix_in[d0, d1, d2, h_new_indexes[d3]]
