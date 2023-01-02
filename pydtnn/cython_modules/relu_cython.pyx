#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-23 Universitat Jaume I
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

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

def relu_cython(x):
    # Warning: the keys in the next dictionary need to be the given strings
    # (i.e., for np.int8, neither np.int8, nor str(np.int8) work as a valid key)
    fake_arg = {'int8': <np.int8_t> 0,
                'float32': <np.float32_t> 0.0,
                'float64': <np.float64_t> 0.0}
    try:
        return relu_cython_template(x, fake_arg[str(x.dtype)])
    except KeyError:
        raise TypeError(f"Type '{x.dtype}' is not supported by relu_cython!")

# The fake argument is used to generate specialized versions of this function
def relu_cython_template(x, npDT fake_arg):
    shape = x.shape
    size = np.prod(shape)
    cdef:
        np.ndarray max = np.zeros((size,), dtype=x.dtype)
        np.ndarray mask = np.zeros((size,), dtype=np.int8)
        npDT[:] x_view = x.reshape(-1)
        npDT[:] max_view = max
        np.int8_t[:] mask_view = mask
    relu_cython_inner(x_view, max_view, mask_view)
    return max.reshape(shape), mask.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner(npDT[:] x,
                       npDT[:] max,
                       np.int8_t[:] mask):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
