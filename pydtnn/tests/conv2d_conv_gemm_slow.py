"""
Unitary tests (exhaustive ones) for Conv2D layer using the convGemm library

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.Conv2DConvGemmSlowTestCase

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.Conv2DConvGemmSlowTestCase

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.Conv2DConvGemmSlowTestCase.test_name
"""

import sys
import unittest

import numpy as np
from rich.console import Console

from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.tests.common import verbose_test
from .conv2d_conv_gemm import Conv2DConvGemmTestCase, D


class Conv2DConvGemmSlowTestCase(Conv2DConvGemmTestCase):
    """
    Tests that Conv2D with conv_gemm leads to the same results than Conv2d with mm and i2c.T (exhaustive version)
    """

    def test_forward_backward_multiple_params(self):
        """Tests that different input matrices, paddings and strides, lead to the same solution"""
        d = D()
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="dots10"):
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
                                                        weights = np.random.rand(d.kn, d.c, d.kh, d.kw).astype(
                                                            np.float32,
                                                            order='C')
                                                        self._test_forward_backward(d, x, weights)


if __name__ == '__main__':
    try:
        Conv2DCPU()
    except NameError:
        sys.exit(-1)
    unittest.main()
