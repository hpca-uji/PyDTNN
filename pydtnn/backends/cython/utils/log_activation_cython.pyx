
cimport cython

from cython.cimports.libc.math import exp, log
from cython.parallel import prange

from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "log_fwd_cython",
    "log_bwd_cython"
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def log_fwd_cython(npDT[::1] x, npDT[::1] y) -> None:
    cdef int i
    # return np.log(1 / (1 + np.exp(-x)))
    #for i in prange(x.shape[0], nogil=True):
    #    y[i] = <npDT> log(1 / ( 1 + exp(-1*x[i])))
    
    # NOTE: Log propierty: "log(a / b) = log(a) - log(b)", and "log(1) = 0 ==> log(1 / b) = -log(b)"
    for i in prange(x.shape[0], nogil=True):
        y[i] = <npDT> ((-1.0) * log( 1.0 + exp(-1.0*x[i])))
        #x[i] = <npDT> ((-1.0) * log( 1.0 + exp(-1.0*x[i])))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def log_bwd_cython(npDT[::1] dy, npDT[::1] dx) -> None:
    cdef int i
    # return 1 / (np.exp(dy) + 1)
    
    for i in prange(dy.shape[0], nogil=True):
        dx[i] = <npDT> (1 / (exp(dy[i]) + 1.0))
        #dy[i] = <npDT> ((-1.0) * log( 1.0 + exp(-1.0*x[i])))

