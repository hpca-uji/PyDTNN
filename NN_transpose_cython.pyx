""" Python Distributed Training of Neural Networks - PyDTNN

PyDTNN is a light-weight library for distributed Deep Learning training and 
inference that offers an initial starting point for interaction with 
distributed training of (and inference with) deep neural networks. PyDTNN 
priorizes simplicity over efficiency, providing an amiable user interface 
which enables a flat accessing curve. To perform the training and inference 
processes, PyDTNN exploits distributed inter-process parallelism (via MPI) 
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

__author__ = "Manuel F. Dolz, Enrique S. Quintana, Sergio Barrachina Mir, " \
             "Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz", "Enrique S. Quintana", "Sergio Barrachina Mir",
               "Mar Catalan", "Adrian Castello"]
__date__ = "2021/01/14"

__email__ = "barrachi@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Sergio Barrachina Mir"
__status__ = "Production"
__version__ = "1.1.0"

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

def transpose_1230_cython(original, transposed):
    """Transpose a 4D matrix from (0,1,2,3) to (1,2,3,0). The outer loop is parallelized."""
    if original.dtype == np.float32:
        transpose_1230_cython_float32(original, transposed)
    else:
        raise ValueError("Type {} not supported for transpose_1230_cython") from None

@cython.boundscheck(False)
@cython.wraparound(False)
cdef transpose_1230_cython_float32(np.ndarray[np.float32_t, ndim=4] original,
                                   np.ndarray[np.float32_t, ndim=4] transposed):
    cdef int d0, d1, d2, d3
    for d0 in prange(original.shape[0], nogil=True, schedule="static"):
        for d1 in range(original.shape[1]):
            for d2 in range(original.shape[2]):
                for d3 in range(original.shape[3]):
                    transposed[d1, d2, d3, d0] = original[d0, d1, d2, d3]

def transpose_1230_2nd_cython(original, transposed):
    """Transpose a 4D matrix from (0,1,2,3) to (1,2,3,0). The second loop is parallelized."""
    if original.dtype == np.float32:
        transpose_1230_2nd_cython_float32(original, transposed)
    else:
        raise ValueError("Type {} not supported for transpose_1230_cython") from None

@cython.boundscheck(False)
@cython.wraparound(False)
cdef transpose_1230_2nd_cython_float32(np.ndarray[np.float32_t, ndim=4] original,
                                       np.ndarray[np.float32_t, ndim=4] transposed):
    cdef int d0, d1, d2, d3
    for d0 in range(original.shape[0]):
        for d1 in prange(original.shape[1], nogil=True, schedule="static"):
            for d2 in range(original.shape[2]):
                for d3 in range(original.shape[3]):
                    transposed[d1, d2, d3, d0] = original[d0, d1, d2, d3]

def transpose_2d_f2c_ji_cython(original, transposed):
    """Transpose a 2D matrix from column order (Fortran) to row order (C). Read for each column (j) all its rows (i)."""
    if original.dtype == np.float32:
        transpose_2d_f2c_ji_cython_float32(original, transposed)
    else:
        raise ValueError("Type {} not supported for transpose_2d_f2c_ji_cython") from None

@cython.boundscheck(False)
@cython.wraparound(False)
cdef transpose_2d_f2c_ji_cython_float32(np.ndarray[np.float32_t, ndim=2] original,
                                        np.ndarray[np.float32_t, ndim=2] transposed):
    cdef int d0, d1
    for d1 in prange(original.shape[1], nogil=True, schedule="static"):
        for d0 in range(original.shape[0]):
            transposed[d0, d1] = original[d0, d1]

def transpose_2d_f2c_ij_cython(original, transposed):
    """Transpose a 2D matrix from column order (Fortran) to row order (C). Read for each row (i) all its columns (j)."""
    if original.dtype == np.float32:
        transpose_2d_f2c_ij_cython_float32(original, transposed)
    else:
        raise ValueError("Type {} not supported for transpose_2d_f2c_ji_cython") from None

@cython.boundscheck(False)
@cython.wraparound(False)
cdef transpose_2d_f2c_ij_cython_float32(np.ndarray[np.float32_t, ndim=2] original,
                                        np.ndarray[np.float32_t, ndim=2] transposed):
    cdef int d0, d1
    for d0 in prange(original.shape[0], nogil=True, schedule="static"):
        for d1 in range(original.shape[1]):
            transposed[d0, d1] = original[d0, d1]
