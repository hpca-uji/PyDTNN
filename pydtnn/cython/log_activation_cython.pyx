__all__ = (
    "log_fwd_cython",
    "log_bwd_cython"
)

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.cimports.libc.math import exp, log

# NOTE: Not recommended to use if the "x" contains "big" negative values.

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def log_fwd_cython(npDT[::1] x, npDT[::1] y) -> None:
    cdef int i
    # return np.log(1 / (1 + np.exp(-x)))
    #for i in prange(x.shape[0], nogil=True):
    #    y[i] = <npDT> log(1 / ( 1 + exp(-1*x[i])))
    
    # NOTE: Log propierty: "log(a / b) = log(a) - log(b)", and "log(1) = 0"
    for i in prange(x.shape[0], nogil=True):
        y[i] = <npDT> ((-1.0) * log( 1.0 + exp(-1.0*x[i])))
# --- END sigmoid_fwd_cython --- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def log_bwd_cython(npDT[::1] dy, npDT[::1] dx) -> None:
    cdef int i
    # return 1 / (np.exp(dy) + 1)
    
    for i in prange(dy.shape[0], nogil=True):
        dx[i] = <npDT> (1 / (exp(dy[i]) + 1.0))
# --- END log_bwd_cython --- #

