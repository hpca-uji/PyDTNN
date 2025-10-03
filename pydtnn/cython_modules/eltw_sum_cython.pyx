import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

__all__ = (
    "eltw_sum_cython",
)

# =================== #
# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# --- END COMMON --- #
# =================== #

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def eltw_sum_cython(npDT[::1] x_acc, npDT[::1] x):

    cdef int i
    for i in prange(x.shape[0], nogil=True):
        x_acc[i] += x[i]
# --- END eltw_sum_cython --- #
