import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

__all__ = (
    "relu_cython",
    "capped_relu_cython",
    "leaky_relu_cython"
)

# Declare fused type npDT (to be used with template functions)
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t

###############################################
#                 relu_cython                 #
###############################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def relu_cython(npDT[::1] x, npDT[::1] max, np.int8_t[::1] mask) -> None:

    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
        else: 
            max[i], mask[i] = 0, 0
# --- END relu_cython --- #

###############################################
#             capped_relu_cython              #
###############################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
# NOTE: If cap = 6, then this is a Relu6.
def capped_relu_cython(npDT[::1] x, npDT[::1] max, np.int8_t[::1] mask, float cap) -> None:
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] >= cap:
            max[i], mask[i] = <npDT> cap, 1
        elif x[i] > 0: # cap > x[i] > 0
            max[i], mask[i] = x[i], 1
        else: #  x[i] <= 0
            max[i], mask[i] = <npDT> 0, 0
# --- END capped_relu_cython --- #

###############################################
#              leaky_relu_cython              #
###############################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def leaky_relu_cython(npDT[::1] x, npDT[::1] max, npDT[::1] mask, float negative_slope) -> None:
    cdef int i
    for i in prange(x.shape[0], nogil=True):
        if x[i] > 0:
            max[i], mask[i] = x[i], 1
        elif x[i] < 0:
            max[i], mask[i] = <npDT> (x[i] * negative_slope), <npDT> negative_slope
        else: #x[i] == 0:
            max[i], mask[i] = 0, 0
# --- END leaky_relu_cython --- #
