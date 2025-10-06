__all__ = (
    "sigmoid_fwd_cython",
    "sigmoid_bwd_cython"
)

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.cimports.libc.math import exp

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def sigmoid_fwd_cython(npDT[::1] x, npDT[::1] y) -> None:
    cdef int i
    

    for i in prange(x.shape[0], nogil=True):
        y[i] = <npDT> (1 / ( 1 + exp(-1*x[i])))
# --- END sigmoid_fwd_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def sigmoid_bwd_cython(npDT[::1] dy, npDT[::1] y, npDT[::1] dx) -> None:
    cdef int i
    
    for i in prange(dy.shape[0], nogil=True):
        dx[i] = dy[i] * (y[i] * (1 - y[i]))

# --- END sigmoid_bwd_cython --- #

