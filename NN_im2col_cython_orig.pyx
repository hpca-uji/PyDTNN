import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from math import floor

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ctypedef fused ndarray4d:
    np.ndarray[np.uint8_t, ndim=4]
    np.ndarray[np.float32_t, ndim=4]
    np.ndarray[np.float64_t, ndim=4]


def im2col_cython(x, 
                  int KH, int KW, 
                  int padding, int stride):
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]

    cdef int HH = floor((H + 2 * padding - KH) / stride) + 1
    cdef int WW = floor((W + 2 * padding - KW) / stride) + 1

    cdef int p = padding
    cdef np.ndarray x_padded = np.pad(x,
            ((0, 0), (0, 0), (p, p), (p, p)), mode='constant').astype(x.dtype)

    cdef np.ndarray cols = np.zeros(
            (C * KH * KW, N * HH * WW), dtype=x.dtype)

    if (x.dtype == np.int8):
        im2col_cython_inner_int8(cols, x_padded, N, C, H, W, HH, WW,
                            KH, KW, padding, stride)
    elif (x.dtype == np.float32):
        im2col_cython_inner_float32(cols, x_padded, N, C, H, W, HH, WW,
                            KH, KW, padding, stride)
    elif (x.dtype == np.float64):
        im2col_cython_inner_float64(cols, x_padded, N, C, H, W, HH, WW,
                            KH, KW, padding, stride)
    else:
        print("Type" + str(x.dtype) + "not supported for im2col_cython!")
        raise

    return cols

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                             np.ndarray[np.int8_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            cols[row, col] = x_padded[n, c, stride * xx + ii, stride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                             np.ndarray[np.float32_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            cols[row, col] = x_padded[n, c, stride * xx + ii, stride * yy + jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                             np.ndarray[np.float64_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    for c in prange(C, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            cols[row, col] = x_padded[n, c, stride * xx + ii, stride * yy + jj]


def col2im_cython(cols, 
                  int N, int C, int H, int W,
                  int KH, int KW, 
                  int padding, int stride):

    cdef int HH = floor((H + 2 * padding - KH) / stride) + 1
    cdef int WW = floor((W + 2 * padding - KW) / stride) + 1

    cdef np.ndarray x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding),
                                         dtype=cols.dtype)
    #cdef np.ndarray x_padded = np.zeros((N, C, H, W), dtype=cols.dtype)

    if (cols.dtype == np.int8):
        col2im_cython_inner_int8(cols, x_padded, N, C, H, W, HH, WW, 
                            KH, KW, padding, stride)
    elif (cols.dtype == np.float32):
        col2im_cython_inner_float32(cols, x_padded, N, C, H, W, HH, WW, 
                            KH, KW, padding, stride)
    elif (cols.dtype == np.float64):
        col2im_cython_inner_float64(cols, x_padded, N, C, H, W, HH, WW, 
                            KH, KW, padding, stride)
    else:
        print("Type" + str(cols.dtype) + "not supported for col2im_cython!")
        raise
    #print("hola")
    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef int col2im_cython_inner_int8(np.ndarray[np.int8_t, ndim=2] cols,
                             np.ndarray[np.int8_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    #for c in prange(C, nogil=True):
    for c in range(C): #, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(padding, HH-padding):
                        for yy in range(padding, WW-padding):
                            col = n * HH * WW + xx * WW + yy
                            #x_padded[n, c, stride * xx + ii, stride * yy + jj] += cols[row, col]
                            x_padded[n, c, stride * xx + ii - 2*padding, stride * yy + jj - 2*padding] += cols[row, col]
                            

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef int col2im_cython_inner_float32(np.ndarray[np.float32_t, ndim=2] cols,
                             np.ndarray[np.float32_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col

    #for c in prange(C, nogil=True):
    for c in range(C): #, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(padding, HH-padding):
                        for yy in range(padding, WW-padding):
                            col = n * HH * WW + xx * WW + yy
                            #x_padded[n, c, stride * xx + ii, stride * yy + jj] += cols[row, col]
                            x_padded[n, c, stride * xx + ii - 2*padding, stride * yy + jj - 2*padding] += cols[row, col]


#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef int col2im_cython_inner_float64(np.ndarray[np.float64_t, ndim=2] cols,
                             np.ndarray[np.float64_t, ndim=4] x_padded,
                             int N, int C, int H, int W, int HH, int WW,
                             int KH, int KW, int padding, int stride) except? -1:
    cdef int c, ii, jj, row, yy, xx, n, col
  
    #for c in prange(C, nogil=True):
    for c in range(C): #, nogil=True):
        for ii in range(KH):
            for jj in range(KW):
                row = c * KH * KW + ii * KW + jj
                for n in range(N):
                    for xx in range(HH):
                        for yy in range(WW):
                            col = n * HH * WW + xx * WW + yy
                            #print(stride * xx + ii - 2*padding, stride * yy + jj - 2*padding)
                            #if False: #(stride * xx + ii) < padding:
                                #print("border pixel", (stride * xx + ii), (stride, xx, ii, padding))
                            #pass
                            x_padded[n, c, stride * xx + ii, stride * yy + jj] += cols[row, col]
