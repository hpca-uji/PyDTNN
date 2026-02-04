import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def bn_training_fwd_cython[T: _npDT](x: _npDT_2Dims[T],
                                     y: _npDT_2Dims[T],
                                     xn: _npDT_2Dims[T],
                                     std: _npDT_1Dims[T],
                                     gamma: _npDT_1Dims[T],
                                     beta: _npDT_1Dims[T],
                                     mean: _npDT_1Dims[T],
                                     var: _npDT_1Dims[T],
                                     eps: float) -> None:
    """
    Args:
        x (npDT_2Dims): The input.
        y (npDT_2Dims): 2-dimensional ndarray where the output is stored.
        xn (npDT_2Dims): 2-dimensional ndarray where the mean of x is stored.
        std (npDT_1Dims): 2-dimensional ndarray where the standard deviation of x is stored.
        gamma (npDT_1Dims): batch normalization gamma.
        beta (npDT_1Dims): batch normalization beta.
        mean (npDT_1Dims): input means.
        var (npDT_1Dims): input variances.
        eps (float): epsilon.

    Returns:
        out: Nothing. The output is stored in \"y\", \"xn\" and \"std\".

    Note:
        It's never used.
    """
    ...
# ---


def bn_training_bwd_cython[T: _npDT](dx: _npDT_2Dims[T],
                                     dy: _npDT_2Dims[T],
                                     xn: _npDT_2Dims[T],
                                     std: _npDT_1Dims[T],
                                     gamma: _npDT_1Dims[T],
                                     dgamma: _npDT_1Dims[T],
                                     dbeta: _npDT_1Dims[T]) -> None:
    """
    Args:
        dx (npDT_2Dims): The 2 dimensional array that contains the gradient of the input forward's (that is the output).
        dy (npDT_2Dims): The 2 dimensional array that contains the gradient of the backward's input.
        xn (npDT_2Dims): The 2 dimensional array that contains the normalized input's value.
        std (npDT_1Dims): The 1 dimensions ndarray that stores the standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions thtat stores the gamma's values
        dgamma (npDT_1Dims): The input's 1 dimensions ndarray the gradient of the gamma's values
        dbeta (npDT_1Dims): The input's 1 dimensions ndarray the  gradient of the beta's values

    Returns:
        Nothing. The output will be stored in "dx".
    """
    ...
