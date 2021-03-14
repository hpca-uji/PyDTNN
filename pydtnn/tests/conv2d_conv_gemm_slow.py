"""
Unitary tests (exhaustive ones) for Conv2D layer using the convGemm library

For running all the tests quietly, execute from the parent directory:
    python -m unittest pydtnn.tests.Conv2DConvGemmSlowTestCase

For running all the tests verbosely, execute from the parent directory:
    python -m unittest -v pydtnn.tests.Conv2DConvGemmSlowTestCase

For running an individual test verbosely, execute from the parent directory:
    python -m unittest -v pydtnn.tests.Conv2DConvGemmSlowTestCase.test_name
"""

import sys
import unittest

import numpy as np

from .conv2d_conv_gemm import Conv2DConvGemmTestCase, D
from .tools import Spinner
from ..layers import Conv2D


def verbose():
    """Returns True if unittest has been called with -v or --verbose options."""
    return '-v' in sys.argv or '--verbose' in sys.argv


class Conv2DConvGemmSlowTestCase(Conv2DConvGemmTestCase):
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
