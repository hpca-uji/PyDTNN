cimport cython
from cython.parallel import prange
from pydtnn.backends.cython.utils.base cimport npDT

__all__ = (
    "depthwise_conv_nhwc_cython",
    "depthwise_conv_backward_nhwc_cython"
)


def div_arrays_set_if_zero(npDT[::1] dividend,
                           npDT[::1] divider,
                           npDT default_value):
    cdef int i, n = dividend.shape[0]

    for i in prange(n, nogil=True):
        dividend[i] = <npDT> (dividend[i] / divider[i] if divider[i] != 0 else default_value)