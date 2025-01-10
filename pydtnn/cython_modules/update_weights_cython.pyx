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




