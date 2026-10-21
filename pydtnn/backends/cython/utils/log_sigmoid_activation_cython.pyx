from cython.cimports.libc.math import exp, log
from cython.parallel import prange

from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "log_sigmoid_fwd_cython",
    "log_sigmoid_bwd_cython"
)


def log_sigmoid_fwd_cython(npDT[::1] x, npDT[::1] y) -> None:
    cdef int i
    # return np.log(1 / (1 + np.exp(-x)))
    #for i in prange(x.shape[0], nogil=True):
    #    y[i] = <npDT> log(1 / ( 1 + exp(-1*x[i])))
    
    # NOTE: Log propierty: "log(a / b) = log(a) - log(b)", and "log(1) = 0 ==> log(1 / b) = -log(b)"
    for i in prange(x.shape[0], nogil=True):
        y[i] = <npDT> ((-1.0) * log( 1.0 + exp(-1.0*x[i])))

def log_sigmoid_bwd_cython(npDT[::1] dy, npDT[::1] dx, y: npDT[::1]) -> None:
    cdef int i

    for i in prange(dy.shape[0], nogil=True):
        dx[i] = <npDT> (dy[i] / (exp(y[i]) + 1.0))

# def log_sigmoid_bwd_cython(npDT[::1] dy, npDT[::1] dx) -> None:
#     cdef int i
#     
#     for i in prange(dy.shape[0], nogil=True):
#         dx[i] = <npDT> (1 / (exp(dy[i]) + 1.0))
