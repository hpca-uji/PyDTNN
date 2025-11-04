import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

__all__ = (
    "depthwise_conv_nhwc_cython",
    "depthwise_conv_backward_nhwc_cython"
)

# =================== #
# --- COMMON --- #
ctypedef fused npDT:
    np.int8_t
    np.float32_t
    np.float64_t
    # NOTE: in order to extend the supported data types, add the new types here.
# -- END npDT -- #
# =================== #

# =============== #
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def div_arrays_set_if_zero(npDT[::1] dividend,
                           npDT[::1] divider,
                           npDT default_value):
    cdef int i, n = dividend.shape[0]

    for i in prange(n, nogil=True):
        dividend[i] = <npDT> (dividend[i] / divider[i] if divider[i] != 0 else default_value)