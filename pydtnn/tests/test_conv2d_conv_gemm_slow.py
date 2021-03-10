"""
Unitary tests (exhaustive ones) for Conv2D layer using the convGemm library

For running all the tests quietly, execute from the parent directory:
    python -m unittest unittests.TestConv2DConvGemmSlow

For running all the tests verbosely, execute from the parent directory:
    python -m unittest -v unittests.TestConv2DConvGemmSlow

For running an individual test verbosely, execute from the parent directory:
    python -m unittest -v unittests.TestConv2DConvGemmSlow.test_name
"""

import sys
import unittest

import numpy as np

from .tools import Spinner

try:
    from NN_layer import Conv2D
    from NN_model import Model
    from NN_conv_gemm import ConvGemm
    from NN_im2col_cython import im2col_cython
    from unittests.test_conv2d_conv_gemm import TestConv2DConvGemm, D, _print_with_header, get_conv2d_layers
except ModuleNotFoundError:
    print("Please, execute as 'python -m unittest unittests.TestConv2DConvGemmSlow'")


def verbose():
    """Returns True if unittest has been called with -v or --verbose options."""
    return '-v' in sys.argv or '--verbose' in sys.argv


class TestConv2DConvGemmSlow(TestConv2DConvGemm):
    """
    Tests that Conv2D with conv_gemm leads to the same results than Conv2d with mm and i2c.T (exhaustive version)
    """

    def test_forward_backward_multiple_params(self):
        """Tests that different input matrices, paddings and strides, lead to the same solution"""
        d = D()
        spinner = Spinner()
        for d.kn in range(1, 4):
            for d.b in range(1, 4):
                for d.c in range(1, 4):
                    for d.h in range(8, 11):
                        for d.w in range(8, 11):
                            for d.kh in range(2, d.h + 1):
                                for d.kw in range(2, d.w + 1):
                                    for d.vpadding in range(0, 4):
                                        for d.hpadding in range(0, 4):
                                            for d.vstride in range(1, 4):
                                                for d.hstride in range(1, 4):
                                                    if not verbose():
                                                        spinner.render()
                                                    if d.b != 1 or d.c != 1:
                                                        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32,
                                                                                                      order='C')
                                                    else:
                                                        x = []
                                                        for i in range(d.h):
                                                            b = (i + 1) * 100
                                                            x = np.concatenate((x, np.arange(b, b + d.w)))
                                                        x = x.reshape((d.b, d.c, d.h, d.w)).astype(np.float32,
                                                                                                   order='C')
                                                    weights = np.random.rand(d.kn, d.c, d.kh, d.kw).astype(np.float32,
                                                                                                           order='C')
                                                    self._test_forward_backward(d, x, weights)
        if not verbose():
            spinner.stop()


if __name__ == '__main__':
    try:
        Conv2D()
    except NameError:
        sys.exit(-1)
    unittest.main()
