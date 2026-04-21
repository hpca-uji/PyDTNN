cimport cython
from cython.parallel import prange
from libc.math cimport sqrt
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "bn_training_fwd_cython",
    "bn_training_bwd_cython"
)

# --- FORWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_training_fwd_cython(npDT[:,::1] x,
                           npDT[:,::1] y,
                           npDT[:,::1] xn,
                           npDT[::1] std,
                           npDT[::1] gamma,
                           npDT[::1] beta,
                           npDT[::1] mean,
                           npDT[::1] var,
                           float eps) -> None:

    cdef int i, j, n, c

    n = x.shape[0]
    c = x.shape[1]

    for i in prange(n, nogil=True):
        for j in range(c):
            std[j] = <npDT> sqrt(var[j] + eps)
            xn[i, j] = <npDT> ((x[i, j] - mean[j]) / std[j])
            y[i, j] = xn[i, j] * gamma[j] + beta[j]
    return None


# --- BACKWARD --- #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def bn_training_bwd_cython(npDT[:, ::1] dx,
                           npDT[:, ::1] dy,
                           npDT[:, ::1] xn,
                           npDT[::1] std,
                           npDT[::1] gamma,
                           npDT[::1] dgamma,
                           npDT[::1] dbeta) -> None:

    cdef int i, j, n = dy.shape[0]

    for i in prange(n, nogil=True, schedule='static'):
        for j in range(dy.shape[1]):
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta) 
            dx[i, j] = <npDT> ((gamma[j] / (std[j] * n)) * (n * dy[i, j] - xn[i, j] * dgamma[j] - dbeta[j]))
# --- bn_training_bwd_cython --- #

