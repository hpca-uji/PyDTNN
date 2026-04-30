import numpy as np

cimport cython
cimport numpy as np

from cython.parallel import prange

__all__ = (
    "bn_inference_cython",
    "bn_inference_nchw_cython",
    "bn_relu_inference_cython"
)

# --- COMMON ---
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.

# --- Base Batch Normalization ---
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_inference_cython(npDT[:, ::1] x, 
                             npDT[:, ::1] y,
                             npDT[::1] running_mean, 
                             npDT[::1] std, 
                             npDT[::1] gamma, 
                             npDT[::1] beta) -> None:
    cdef int i, j

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            y[i, j] = <npDT> (gamma[j] * (x[i, j] - running_mean[j]) / std[j]) + beta[j]




# --- NCHW Batch Normalization ---
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_inference_nchw_cython(npDT[:, ::1] x, 
                             npDT[:, ::1] y,
                             npDT[::1] running_mean, 
                             npDT[::1] std, 
                             npDT[::1] gamma, 
                             npDT[::1] beta) -> None:

    cdef int i, j

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
                    y[i, j] = <npDT> (gamma[j] * (x[i, j] - running_mean[j]) / std[j]) + beta[j]







@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_relu_inference_cython(npDT[:, ::1] x,
                             npDT[:, ::1] y,
                             npDT[::1] running_mean,
                             npDT[::1] inv_std,
                             npDT[::1] gamma,
                             npDT[::1] beta) -> None:
    #   xn = (x - self.running_mean) * inv_std
    #   y = gamma * xn + beta
    
    # cdef np.ndarray[npDT, ndim=2] y = np.zeros_like(x, dtype=x.dtype)

    
    cdef int i, j = 0
    cdef npDT tmp

    for i in prange(x.shape[0], nogil=True, schedule='static'):
        for j in range(x.shape[1]):
            tmp = (x[i, j] - running_mean[j]) * inv_std[j]
            y[i, j] = max((tmp * gamma[j]) + beta[j], 0)

