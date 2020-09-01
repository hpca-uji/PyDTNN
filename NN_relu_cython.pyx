""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
çprocesses, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors and GPUs at node level. For that, PyDTNN 
uses MPI4Py for message-passing, BLAS calls via NumPy for multicore processors
and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ =  "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"


import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from math import floor

def relu_cython(x):
    #if not x.flags['C_CONTIGUOUS']:
    #    np.ascontiguousarray(x, dtype=np.float32)
    shape = x.shape
    cdef np.ndarray max = np.zeros((np.prod(shape)), dtype=x.dtype)
    cdef np.ndarray mask = np.zeros((np.prod(shape)), dtype=np.int8)

    if (x.dtype == np.int8):
        relu_cython_inner_int8(x.reshape(-1), max, mask)
    elif (x.dtype == np.float32):
        relu_cython_inner_float32(x.reshape(-1), max, mask)
    elif (x.dtype == np.float64):
        relu_cython_inner_float64(x.reshape(-1), max, mask)
    else:
        print("Type %s not supported for relu_cython!" % (str(x.dtype)))
        raise

    return max.reshape(shape), mask.reshape(shape)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner_int8(np.ndarray[np.int8_t, ndim=1] x, 
                              np.ndarray[np.int8_t, ndim=1] max,
                              np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i, j=0
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner_float32(np.ndarray[np.float32_t, ndim=1] x, 
                              np.ndarray[np.float32_t, ndim=1] max,
                              np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef relu_cython_inner_float64(np.ndarray[np.float64_t, ndim=1] x, 
                              np.ndarray[np.float64_t, ndim=1] max,
                              np.ndarray[np.int8_t, ndim=1] mask):
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
