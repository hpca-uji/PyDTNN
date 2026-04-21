cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "pointwise_conv_cython",
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def pointwise_conv_cython(npDT[:,:,:,::1] x, npDT[:,::1] k, npDT[:,:,:,::1] out) -> None:

    cdef int n = x.shape[0]
    cdef int c = x.shape[1]
    cdef int h = x.shape[2]
    cdef int w = x.shape[3]

    cdef int co = k.shape[0]

    #cdef npDT[:,:,:,::1] out = np.zeros((n, co, h, w))

    cdef int nn, cco, cc, ii, jj

    for cco in prange(co, nogil=True):
        for cc in range(c):
            for nn in range(n):
                for ii in range(h):
                    for jj in range(w):
                        out[nn, cco, ii, jj] += x[nn, cc, ii, jj] * k[cco, cc]
