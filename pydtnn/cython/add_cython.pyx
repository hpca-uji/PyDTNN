import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

__all__ = (
    "add_cython",
)

# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def add_cython(npDT[:,::1] x, npDT[::1] b) -> None:
    cdef int i, j

    for i in prange(x.shape[0], nogil=True):
        for j in range(x.shape[1]):
            x[i, j] += b[i]
# --- END add_cython --- #