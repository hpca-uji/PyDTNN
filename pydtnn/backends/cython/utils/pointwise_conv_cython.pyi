
from pydtnn.backends.cython.utils.base import _npDT, _npDT_2Dims, _npDT_4Dims

def pointwise_conv_cython[T: _npDT](x: _npDT_4Dims[T], k: _npDT_2Dims[T], out: _npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): 4-dimensinal array where the input data is stored.
        k (npDT_2Dims): 2-dimensinal array where the kernel is stored.
        out (npDT_4Dims): 4-dimensinal array where the output is stored.
    Returns:
        Nothing. The output is stored in `out`.
    """
