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

# This code has been inspired from cthorey, see:
#    https://github.com/cthorey/CS231/blob/master/assignment2/cs231n/im2col_cython.pyx

def im2col_cython(x, 
                  int KH, int KW, 
                  int vpadding, int hpadding, int vstride, int hstride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    cdef int HH = floor((H + 2 * vpadding - KH) / vstride) + 1
    cdef int WW = floor((W + 2 * hpadding - KW) / hstride) + 1

    cdef np.ndarray x_padded = np.pad(x,
            ((0, 0), (0, 0), (vpadding, vpadding), (hpadding, hpadding)), mode='constant').astype(x.dtype)

    cdef np.ndarray cols = np.zeros(
            (C * KH * KW, N * HH * WW), dtype=x.dtype)

    if (x.dtype == np.int8):
        im2col_cython_inner_int8(cols, x_padded, N, C, H, W, HH, WW,
                                 KH, KW, vstride, hstride)
    elif (x.dtype == np.float32):
        im2col_cython_inner_float32(cols, x_padded, N, C, H, W, HH, WW,
                                 KH, KW, vstride, hstride)
    elif (x.dtype == np.float64):
        im2col_cython_inner_float64(cols, x_padded, N, C, H, W, HH, WW,
                                 KH, KW, vstride, hstride)
    else:
        print("Type %s not supported for im2col_cython!" % (str(cols.dtype)))
        raise

    return cols

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                             np.ndarray[np.int8_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int vstride, int hstride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            cols[row, col] = x_padded[n, c, vstride * xx + ii, hstride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                             np.ndarray[np.float32_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int vstride, int hstride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            cols[row, col] = x_padded[n, c, vstride * xx + ii, hstride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                             np.ndarray[np.float64_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int vstride, int hstride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            cols[row, col] = x_padded[n, c, vstride * xx + ii, hstride * yy + jj]

def col2im_cython(cols, 
                  int N, int C, int H, int W,
                  int KH, int KW, 
                  int vpadding, int hpadding, int vstride, int hstride):

    cdef int HH = floor((H + 2 * vpadding - KH) / vstride) + 1
    cdef int WW = floor((W + 2 * hpadding - KW) / hstride) + 1

    cdef np.ndarray x_padded = np.zeros((N, C, H + 2 * vpadding, W + 2 * hpadding),
                                         dtype=cols.dtype)
    if (cols.dtype == np.int8):
        col2im_cython_inner_int8(cols, x_padded, N, C, H, W, HH, WW, 
                                 KH, KW, vstride, hstride)
    elif (cols.dtype == np.float32):
        col2im_cython_inner_float32(cols, x_padded, N, C, H, W, HH, WW, 
                                 KH, KW, vstride, hstride)
    elif (cols.dtype == np.float64):
        col2im_cython_inner_float64(cols, x_padded, N, C, H, W, HH, WW, 
                                 KH, KW, vstride, hstride)
    else: 
        print("Type %s not supported for col2im_cython!" % (str(cols.dtype)))
        raise

    if vpadding > 0 or hpadding > 0:
        return x_padded[:, :, vpadding:-vpadding, hpadding:-hpadding]
    return x_padded

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                             np.ndarray[np.int8_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int vstride, int hstride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            x_padded[n, c, vstride * xx + ii, hstride * yy + jj] += cols[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                             np.ndarray[np.float32_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int vstride, int hstride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            x_padded[n, c, vstride * xx + ii, hstride * yy + jj] += cols[row, col]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int col2im_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                             np.ndarray[np.float64_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int vstride, int hstride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col
  
    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            x_padded[n, c, vstride * xx + ii, hstride * yy + jj] += cols[row, col]
