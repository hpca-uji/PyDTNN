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
def intersect_2d_indexes_cython(np.ndarray local_rows,
                                np.ndarray local_cols,
                                np.ndarray global_rows,
                                np.ndarray global_cols):
    
    cdef int count = 0
    cdef int i_local_row = 0
    cdef int i_global_row = 0
    cdef int max_size = min(len(local_rows), len(global_rows))
    cdef np.ndarray[np.int32_t, ndim=1] intersected_rows = np.empty(max_size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] intersected_cols = np.empty(max_size, dtype=np.int32)

    while i_local_row < len(local_rows) and i_global_row < len(global_rows):
        local_row = local_rows[i_local_row]
        global_row = global_rows[i_global_row]
        if local_row < global_row:
            i_local_row += 1
        elif local_row > global_row:
            i_global_row += 1
        else:
            local_col = local_cols[i_local_row]
            global_col = global_cols[i_global_row]
            if local_col == global_col:
                intersected_rows[count] = local_row
                intersected_cols[count] = local_col
                count += 1
            i_local_row += 1
            i_global_row += 1
    return intersected_rows[:count], intersected_cols[:count]

