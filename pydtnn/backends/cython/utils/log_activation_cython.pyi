
from pydtnn.backends.cython.utils.base import _npDT, _npDT_1Dims

def log_fwd_cython[T:_npDT](x: _npDT_1Dims[T], y: _npDT_1Dims[T]) -> None:
    """
    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        y (npDT_1Dims): 1-dimensional array where the ouput is stored
    Returns:
        Nothing. The output is stored in "y".
    """


def log_bwd_cython[T:_npDT](dy: _npDT_1Dims[T], dx: _npDT_1Dims[T]) -> None:
    """
    Args:
        dy (npDT_1Dims): 1-dimensional input's array.
        dx (npDT_1Dims): 1-dimensional array where the output will be stored.
    Returns:
        Nothing. The output is stored in "dx".
    """
