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
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def intersect_2d_indexes_cython(cnp.ndarray[cnp.int_t, ndim=1] local_rows,
                                cnp.ndarray[cnp.int_t, ndim=1] local_cols,
                                cnp.ndarray[cnp.int_t, ndim=1] global_rows,
                                cnp.ndarray[cnp.int_t, ndim=1] global_cols):
    
    dtype = [('row', 'int32'), ('col', 'int32')]
    
    cdef cnp.ndarray local_indices = np.array(list(zip(local_rows, local_cols)), dtype=dtype)
    cdef cnp.ndarray global_indices = np.array(list(zip(global_rows, global_cols)), dtype=dtype)
    
    cdef cnp.ndarray intersection_indices = np.intersect1d(local_indices, global_indices, assume_unique=False)
    
    intersection_rows = intersection_indices['row']
    intersection_cols = intersection_indices['col']

    return intersection_rows, intersection_cols