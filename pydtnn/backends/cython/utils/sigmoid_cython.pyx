cimport cython
from cython.parallel import prange
from cython.cimports.libc.math import exp
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "sigmoid_fwd_cython",
    "sigmoid_bwd_cython"
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def sigmoid_fwd_cython(npDT[::1] x, npDT[::1] y) -> None:
    cdef int i
    

    for i in prange(x.shape[0], nogil=True):
        y[i] = <npDT> (1 / ( 1 + exp(-1*x[i])))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def sigmoid_bwd_cython(npDT[::1] dy, npDT[::1] y, npDT[::1] dx) -> None:
    cdef int i
    
    for i in prange(dy.shape[0], nogil=True):
        dx[i] = dy[i] * (y[i] * (1 - y[i]))


