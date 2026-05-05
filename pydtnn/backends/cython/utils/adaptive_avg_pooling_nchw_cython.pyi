from pydtnn.backends.cython.utils.base import _npDT, _npDT_4Dims

def adaptive_avg_pooling_fwd_nchw_cython[T: _npDT](x: _npDT_4Dims[T], pooled_x: _npDT_4Dims[T]) -> None:
    """
    Args:
        x (npDT_4Dims): data input.
        pooled_x (npDT_4Dims): ndarray where the output will be stored. It must be filled with zeros.
    Returns:
        Nothing; the return is stored in "dx".
    """

def adaptive_avg_pooling_bwd_nchw_cython[T: _npDT](dy: _npDT_4Dims[T], dx: _npDT_4Dims[T]) -> None:
    """
    Args:
        dy (npDT_4Dims): data input.
        dx (npDT_4Dims): ndarray where the output will be stored. It must be filled with zeros.
    Returns:
        Nothing; the return is stored in "dx".
    """
