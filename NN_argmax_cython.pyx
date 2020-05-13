""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
for clusters and intra-process (via multi-threading) parallelism to leverage 
the presence of multicore processors at node level.

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
__version__ = "1.0.1"


import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from math import floor

def argmax_cython(x, axis=0):
    if axis == 0: x = x.T
    #if not x.flags['C_CONTIGUOUS']:
    #    np.ascontiguousarray(x, dtype=np.float32)
    cdef np.ndarray amax = np.zeros((x.shape[0]), dtype=np.int32)

    if (x.dtype == np.int8):
        argmax_cython_inner_int8(x, amax)
    elif (x.dtype == np.float32):
        argmax_cython_inner_float32(x, amax)
    elif (x.dtype == np.float64):
        argmax_cython_inner_float64(x, amax)
    else:
        print("Type %s not supported for argmax_cython!" % (str(x.dtype)))
        raise

    return amax

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] x, 
                              np.ndarray[np.int32_t, ndim=1] amax):
    cdef int i, j, idx_maxval
    cdef np.int8_t maxval, minval
    minval = np.finfo(np.int8).min

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i,j] > maxval:
                maxval, idx_maxval = x[i,j], j
        amax[i] = idx_maxval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] x, 
                              np.ndarray[np.int32_t, ndim=1] amax):
    cdef int i, j, idx_maxval
    cdef np.float32_t maxval, minval
    minval = np.finfo(np.float32).min

    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i,j] > maxval:
                maxval, idx_maxval = x[i,j], j
        amax[i] = idx_maxval

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] x, 
                              np.ndarray[np.int32_t, ndim=1] amax):
    cdef int i, j, idx_maxval
    cdef np.float64_t maxval, minval
    minval = np.finfo(np.float64).min
    
    for i in prange(x.shape[0], nogil=True):
        maxval, idx_maxval = minval, 0
        for j in range(x.shape[1]):
            if x[i,j] > maxval:
                maxval, idx_maxval = x[i,j], j
        amax[i] = idx_maxval
