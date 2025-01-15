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

    # acc = residuals + (learning_rate * dw)

    cdef int i, j

    for i in prange(dw.shape[0], nogil=True):
        for j in range(dw.shape[1]):
            dw[i, j] = residuals[i, j] + (learning_rate * dw[i, j])
    
    return dw



