"""
Unitary tests for NN_gemm_conv.py.

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.ConvGemmNHWCTestCase

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.ConvGemmNHWCTestCase

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.ConvGemmNHWCTestCase.test_name
"""

import inspect
import math
import sys
import unittest

import numpy as np
from rich.console import Console

from pydtnn.backends.cpu.libs import ConvGemm
from pydtnn.tests.common import verbose_test, D, alexnet_layers
from .tools import print_with_header
from ..cython_modules import im2row_nhwc_cython


def _conv_gemm_and_im2row_mm(weights, x, biases=None, vpadding=0, hpadding=0, vstride=1, hstride=1,
                             vdilation=1, hdilation=1):
    if verbose_test():
        print()
    c, kh, kw, kn = weights.shape
    # b, c, h, w = x.shape
    conv_gemm = ConvGemm(debug=verbose_test())
    cg_biases = biases.copy() if biases is not None else None
    conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x, biases=cg_biases,
                                                vpadding=vpadding, hpadding=hpadding,
                                                vstride=vstride, hstride=hstride,
                                                vdilation=vdilation, hdilation=hdilation)
    conv_gemm_result = conv_gemm_result.reshape(-1, kn)
    x_c = im2row_nhwc_cython(x, kh, kw, vpadding, hpadding, vstride, hstride, vdilation, hdilation)
    w_c = weights.reshape(-1, kn)
    if biases is None:
        im2row_mm_result = x_c @ w_c
    else:
        im2row_mm_result = x_c @ w_c + biases.reshape(-1, kn)
    if verbose_test():
        print_with_header("{} conv_gemm_result".format(inspect.stack()[1][3]), conv_gemm_result)
        print("Shape: ", conv_gemm_result.shape,
              " Sum: ", conv_gemm_result.sum(),
              " Min: ", conv_gemm_result.min(),
              " Max: ", conv_gemm_result.max())
        print_with_header("{} im2row_mm_result".format(inspect.stack()[1][3]), im2row_mm_result)
        print("Shape: ", im2row_mm_result.shape,
              " Sum: ", im2row_mm_result.sum(),
              " Min: ", im2row_mm_result.min(),
              " Max: ", im2row_mm_result.max())
        print("---")
        print("Maximum difference: ",
              max([abs(x - y) for x, y in zip(conv_gemm_result.flatten(), im2row_mm_result.flatten())]))
        print("---")
    return conv_gemm_result, im2row_mm_result


class ConvGemmNHWCTestCase(unittest.TestCase):
    """
    Tests that conv_gemm leads to the same results as i2c and mm.
    """

    def test_raise_on_different_strides(self):
        d = D()
        weights = np.ones((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = np.ones((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
                                                                      vpadding=d.vpadding, hpadding=d.hpadding,
                                                                      vstride=1, hstride=2,
                                                                      vdilation=d.vdilation, hdilation=d.hdilation)
        self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    # def test_handmade_array(self):
    #     """
    #     Test that manual matrices lead to the same solution
    #     """
    #     x = np.array([[[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]]]]).astype(np.float32, order='C')
    #     weights = np.array([[[[1, 1],
    #                           [1, 1]]]]).astype(np.float32, order='C')
    #     padding = 0
    #     stride = 1
    #     dilation = 1
    #     conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
    #                                                                   vpadding=padding, hpadding=padding,
    #                                                                   vstride=stride, hstride=stride,
    #                                                                   vdilation=dilation, hdilation=dilation)
    #     if verbose_test():
    #         print(["{:b}  ".format(int(x)) for x in conv_gemm_result.ravel()])
    #         print(["{:b}  ".format(int(x)) for x in im2row_mm_result.ravel()])
    #     self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    # def test_handmade_array_kn_2(self):
    #     """
    #     Test that manual matrices with kn = 2 lead to the same solution
    #     """
    #     x = np.array([[[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]]]]).astype(np.float32, order='C')
    #     weights = np.array([[[[1, 1],
    #                           [1, 1]]],
    #                         [[[2, 2],
    #                           [2, 2]]]]).astype(np.float32, order='C')
    #     padding = 0
    #     stride = 1
    #     dilation = 1
    #     conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
    #                                                                   vpadding=padding, hpadding=padding,
    #                                                                   vstride=stride, hstride=stride,
    #                                                                   vdilation=dilation, hdilation=dilation)
    #     if verbose_test():
    #         print(["{:b}  ".format(int(x)) for x in conv_gemm_result.ravel()])
    #         print(["{:b}  ".format(int(x)) for x in im2row_mm_result.ravel()])
    #     self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    # def test_handmade_array_kn_2_c_2(self):
    #     """
    #     Test that manual matrices with kn = 2 lead to the same solution
    #     """
    #     x = np.array([[[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]],
    #                    [[1, 2, 4, 8],
    #                     [16, 32, 64, 128]]]]).astype(np.float32, order='C')
    #     weights = np.array([[[[1, 2],
    #                           [4, 8]],
    #                          [[2, 2],
    #                           [2, 2]]],
    #                         [[[4, 4],
    #                           [4, 4]],
    #                          [[8, 8],
    #                           [8, 8]]]]).astype(np.float32, order='C')
    #     padding = 0
    #     stride = 1
    #     dilation = 1
    #     conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
    #                                                                   vpadding=padding, hpadding=padding,
    #                                                                   vstride=stride, hstride=stride,
    #                                                                   vdilation=dilation, hdilation=dilation)
    #     if verbose_test():
    #         print(["{:b}  ".format(int(x)) for x in conv_gemm_result.ravel()])
    #         print(["{:b}  ".format(int(x)) for x in im2row_mm_result.ravel()])
    #     self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    # def test_handmade_array_with_biases(self):
    #     """
    #     Test that manual matrices including b lead to the same solution
    #     """
    #     x = np.array([[[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]]]]).astype(np.float32, order='C')
    #     weights = np.array([[[[1, 1],
    #                           [1, 1]]]]).astype(np.float32, order='C')
    #     biases = np.array([[1024, 2048, 4196]]).astype(np.float32, order='C')
    #     padding = 0
    #     stride = 1
    #     dilation = 1
    #     conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x, biases=biases,
    #                                                                   vpadding=padding, hpadding=padding,
    #                                                                   vstride=stride, hstride=stride,
    #                                                                   vdilation=dilation, hdilation=dilation)
    #     if verbose_test():
    #         print(["{:b}  ".format(int(x)) for x in conv_gemm_result.ravel()])
    #         print(["{:b}  ".format(int(x)) for x in im2row_mm_result.ravel()])
    #     self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    # def test_larger_handmade_array(self):
    #     """
    #     Test that larger manual matrices lead to the same solution
    #     """
    #     x = np.array([[[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]],
    #                    [[128, 256, 512, 1024],
    #                     [2048, 4096, 8192, 16384]]]]).astype(np.float32, order='C')
    #     weights = np.array([[[[1, 2],
    #                           [3, 4]],
    #                          [[4, 5],
    #                           [6, 7]],
    #                          ]]).astype(np.float32, order='C')
    #     padding = 0
    #     stride = 1
    #     dilation = 1
    #     conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
    #                                                                   vpadding=padding, hpadding=padding,
    #                                                                   vstride=stride, hstride=stride,
    #                                                                   vdilation=dilation, hdilation=dilation)
    #     if verbose_test():
    #         print(["{:b}  ".format(int(x)) for x in conv_gemm_result.ravel()])
    #         print(["{:b}  ".format(int(x)) for x in im2row_mm_result.ravel()])
    #     self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    # def test_even_larger_handmade_array(self):
    #     """
    #     Test that even larger manual matrices lead to the same solution
    #     """
    #     x = np.array([[[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]],
    #                    [[128, 256, 512, 1024],
    #                     [2048, 4096, 8192, 16384]]],
    #                   [[[1, 2, 4, 8],
    #                     [16, 32, 64, 128]],
    #                    [[128, 256, 512, 1024],
    #                     [2048, 4096, 8192, 16384]]],
    #                   ]).astype(np.float32, order='C')
    #     weights = np.array([[[[1, 1],
    #                           [1, 1]],
    #                          [[4, 4],
    #                           [4, 4]]]]).astype(np.float32, order='C')
    #     padding = 0
    #     stride = 1
    #     dilation = 1
    #     conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
    #                                                                   vpadding=padding, hpadding=padding,
    #                                                                   vstride=stride, hstride=stride,
    #                                                                   vdilation=dilation, hdilation=dilation)
    #     if verbose_test():
    #         print(["{:b}  ".format(int(x)) for x in conv_gemm_result.ravel()])
    #         print(["{:b}  ".format(int(x)) for x in im2row_mm_result.ravel()])
    #     self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    def test_defaults_with_ones(self):
        """
        Test that the default parameters on ones matrices lead to the same solution
        """
        d = D()
        weights = np.ones((d.c, d.kh, d.kw, d.kn)).astype(np.float32, order='C')
        x = np.ones((d.b, d.h, d.w, d.c)).astype(np.float32, order='C')
        conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
                                                                      vpadding=d.vpadding, hpadding=d.hpadding,
                                                                      vstride=d.vstride, hstride=d.hstride,
                                                                      vdilation=d.vdilation, hdilation=d.hdilation)
        self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    def test_defaults_with_random(self):
        """
        Test that the default parameters on random matrices lead to the same solution
        """
        d = D()
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x,
                                                                      vpadding=d.vpadding, hpadding=d.hpadding,
                                                                      vstride=d.vstride, hstride=d.hstride,
                                                                      vdilation=d.vdilation, hdilation=d.hdilation)
        # if verbose_test():
        #     print("Result[0, 0, 0, 1]=")
        #     partial_l = x[0, 0, 0:d.kh, 1:d.kw+1].flatten()
        #     print(w.flatten() @ partial_l)
        #     print("Result[0, 0, 0, 2]=")
        #     partial_l = x[0, 0, 0:d.kh, 2:d.kw+2].flatten()
        #     print(w.flatten() @ partial_l)
        #     print("Result[0, 0, 1, 0]=")
        #     partial_l = x[0, 0, 1:d.kh+1, 0:d.kw].flatten()
        #     print(w.flatten() @ partial_l)

        self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    def test_defaults_including_biases_with_random(self):
        """
        Test that the default parameters on random matrices, including b, lead to the same solution
        """
        d = D()
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        ho = (d.h + 2 * d.vpadding - d.vdilation * (d.kh - 1) - 1) // d.vstride + 1
        wo = (d.w + 2 * d.hpadding - d.hdilation * (d.kw - 1) - 1) // d.hstride + 1
        biases = np.random.rand(d.b, ho, wo, d.kn).astype(np.float32, order='C')
        conv_gemm_result, im2row_mm_result = _conv_gemm_and_im2row_mm(weights, x, biases=biases,
                                                                      vpadding=d.vpadding, hpadding=d.hpadding,
                                                                      vstride=d.vstride, hstride=d.hstride,
                                                                      vdilation=d.vdilation, hdilation=d.hdilation)
        self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result))

    def test_with_different_kn(self):
        d = D()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print(" kn   Maximum difference    sum(cg_result)")
            print("----+--------------------+-----------------")
        conv_gemm = ConvGemm(debug=False)
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for kn in range(1, 32):
                weights = np.random.rand(d.c, d.kh, d.kw, kn).astype(np.float32, order='C')
                conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                            vpadding=d.vpadding, hpadding=d.hpadding,
                                                            vstride=d.vstride, hstride=d.hstride,
                                                            vdilation=d.vdilation, hdilation=d.hdilation)
                conv_gemm_result = conv_gemm_result.reshape(-1, kn)
                x_c = im2row_nhwc_cython(x, d.kh, d.kw, d.vpadding, d.hpadding,
                                         d.vstride, d.hstride, d.vdilation, d.hdilation)
                w_c = weights.reshape(-1, kn)
                im2row_mm_result = x_c @ w_c
                if verbose_test():
                    print("{:3}    {:9.7f}             {:11.2f}"
                          "".format(kn, max([abs(x - y) for x, y in zip(conv_gemm_result.flatten(),
                                                                        im2row_mm_result.flatten())]),
                                    np.sum(conv_gemm_result)))
                np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(conv_gemm_result,
                                                                                        im2row_mm_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_b(self):
        d = D()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  b   Maximum difference")
            print("----+--------------------")
        conv_gemm = ConvGemm(debug=False)
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for b in range(1, 32):
                x = np.random.rand(b, d.h, d.w, d.c).astype(np.float32, order='C')
                conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                            vpadding=d.vpadding, hpadding=d.hpadding,
                                                            vstride=d.vstride, hstride=d.hstride,
                                                            vdilation=d.vdilation, hdilation=d.hdilation)
                conv_gemm_result = conv_gemm_result.reshape(-1, d.kn)
                x_c = im2row_nhwc_cython(x, d.kh, d.kw, d.vpadding, d.hpadding,
                                         d.vstride, d.hstride, d.vdilation, d.hdilation)
                w_c = weights.reshape(-1, d.kn)
                im2row_mm_result = x_c @ w_c
                if verbose_test():
                    print("{:3}    {:9.7f}".format(b,
                                                   max([abs(x - y) for x, y
                                                        in
                                                        zip(conv_gemm_result.flatten(), im2row_mm_result.flatten())])))
                np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(conv_gemm_result,
                                                                                        im2row_mm_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_padding(self):
        d = D()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  p   Maximum difference")
            print("----+--------------------")
        conv_gemm = ConvGemm(debug=False)
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for padding in range(0, 5):
                conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                            vpadding=padding, hpadding=padding,
                                                            vstride=d.vstride, hstride=d.hstride,
                                                            vdilation=d.vdilation, hdilation=d.hdilation)
                conv_gemm_result = conv_gemm_result.reshape(-1, d.kn)
                x_c = im2row_nhwc_cython(x, d.kh, d.kw, padding, padding,
                                         d.vstride, d.hstride, d.vdilation, d.hdilation)
                w_c = weights.reshape(-1, d.kn)
                im2row_mm_result = x_c @ w_c
                if verbose_test():
                    print("{:3}    {:9.7f}".format(padding,
                                                   max([abs(x - y) for x, y
                                                        in
                                                        zip(conv_gemm_result.flatten(), im2row_mm_result.flatten())])))
                np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(conv_gemm_result,
                                                                                        im2row_mm_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_stride(self):
        d = D()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  s   Maximum difference")
            print("----+--------------------")
        conv_gemm = ConvGemm(debug=False)
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for stride in range(1, 6):
                conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                            vpadding=d.vpadding, hpadding=d.hpadding,
                                                            vstride=stride, hstride=stride,
                                                            vdilation=d.vdilation, hdilation=d.hdilation)
                conv_gemm_result = conv_gemm_result.reshape(-1, d.kn)
                x_c = im2row_nhwc_cython(x, d.kh, d.kw, d.vpadding, d.hpadding, stride, stride,
                                         d.vdilation, d.hdilation)
                w_c = weights.reshape(-1, d.kn)
                im2row_mm_result = x_c @ w_c
                if verbose_test():
                    print("{:3}    {:9.7f}".format(stride,
                                                   max([abs(x - y) for x, y
                                                        in
                                                        zip(conv_gemm_result.flatten(), im2row_mm_result.flatten())])))
                np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(conv_gemm_result,
                                                                                        im2row_mm_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_with_different_strides(self):
        d = D()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print(" vs  hs   Maximum difference")
            print("--------+--------------------")
        conv_gemm = ConvGemm(debug=False)
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for vstride in range(1, 5):
                for hstride in range(1, 5):
                    if vstride == hstride:
                        continue
                    conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                                vpadding=d.vpadding, hpadding=d.hpadding,
                                                                vstride=vstride, hstride=hstride,
                                                                vdilation=d.vdilation, hdilation=d.hdilation)
                    conv_gemm_result = conv_gemm_result.reshape(-1, d.kn)
                    x_c = im2row_nhwc_cython(x, d.kh, d.kw, d.vpadding, d.hpadding, vstride, hstride,
                                             d.vdilation, d.hdilation)
                    w_c = weights.reshape(-1, d.kn)
                    im2row_mm_result = x_c @ w_c
                    if verbose_test():
                        print("{:3} {:3}    {:9.7f}".format(vstride, hstride,
                                                            max([abs(x - y) for x, y
                                                                 in
                                                                 zip(conv_gemm_result.flatten(),
                                                                     im2row_mm_result.flatten())])))
                    self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result),
                                    f"Results differ with vstride {vstride} and hstride {hstride}")

    def test_with_different_dilation(self):
        d = D()
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print("  s   Maximum difference")
            print("----+--------------------")
        conv_gemm = ConvGemm(debug=False)
        weights = np.random.rand(d.c, d.kh, d.kw, d.kn).astype(np.float32, order='C')
        x = np.random.rand(d.b, d.h, d.w, d.c).astype(np.float32, order='C')
        np_all_close_for_all_cases = True
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for dilation in range(1, 3):
                conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                            vpadding=d.vpadding, hpadding=d.hpadding,
                                                            vstride=d.vstride, hstride=d.hstride,
                                                            vdilation=dilation, hdilation=dilation)
                conv_gemm_result = conv_gemm_result.reshape(-1, d.kn)
                x_c = im2row_nhwc_cython(x, d.kh, d.kw, d.vpadding, d.hpadding, d.vstride, d.hstride,
                                         dilation, dilation)
                w_c = weights.reshape(-1, d.kn)
                im2row_mm_result = x_c @ w_c
                if verbose_test():
                    print("{:3}    {:9.7f}".format(dilation,
                                                   max([abs(x - y) for x, y
                                                        in
                                                        zip(conv_gemm_result.flatten(), im2row_mm_result.flatten())])))
                np_all_close_for_all_cases = np_all_close_for_all_cases and np.allclose(conv_gemm_result,
                                                                                        im2row_mm_result)
        self.assertTrue(np_all_close_for_all_cases)

    def test_alexnet_layers(self):
        if verbose_test():
            print_with_header("{}".format(inspect.stack()[1][3]), None)
            print(" layer   Maximum difference")
            print("-------+--------------------")
        layers = alexnet_layers
        conv_gemm = ConvGemm(debug=False)
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="bouncingBar"):
            for n, layer in enumerate(layers):
                weights = np.random.rand(layer.c, layer.kh, layer.kw, layer.kn).astype(np.float32, order='C')
                x = np.random.rand(layer.b, layer.h, layer.w, layer.c).astype(np.float32, order='C')
                conv_gemm_result = conv_gemm.conv_gemm_nhwc(weights, x,
                                                            vpadding=layer.vpadding, hpadding=layer.hpadding,
                                                            vstride=layer.vstride, hstride=layer.hstride,
                                                            vdilation=layer.vdilation, hdilation=layer.hdilation)
                conv_gemm_result = conv_gemm_result.reshape(-1, layer.kn)
                x_c = im2row_nhwc_cython(x, layer.kh, layer.kw, layer.vpadding, layer.hpadding,
                                         layer.vstride, layer.hstride, layer.vdilation, layer.hdilation)
                w_c = weights.reshape(-1, layer.kn)
                im2row_mm_result = x_c @ w_c
                if verbose_test():
                    print("   {:2}      {:9.7f}".format(n,
                                                        max([abs(x - y) for x, y
                                                             in
                                                             zip(conv_gemm_result.flatten(),
                                                                 im2row_mm_result.flatten())])))
                    if n == 9:
                        print("Flags for last conv_gemm_result output:")
                        print(conv_gemm_result.flags)
                self.assertTrue(np.allclose(conv_gemm_result, im2row_mm_result),
                                f"Results differ for AlexNet Cifar and ImageNet layers number {n}")


#     def test_deconv_gemm_with_alexnet_layers(self):
#         if verbose_test():
#             print_with_header("{}".format(inspect.stack()[1][3]), None)
#             print(" layer   Maximum difference")
#             print("-------+--------------------")
#         layers = alexnet_layers
#         conv_gemm = ConvGemm(debug=False)
#         console = Console(force_terminal=not verbose_test())
#         with console.status("", spinner="bouncingBar"):
#             for n, layer in enumerate(layers):
#                 weights = np.random.rand(layer.c, layer.kh, layer.kw, layer.kn).astype(np.float32, order='C')
#                 dy = np.random.rand(layer.b, layer.ho, layer.wo, layer.kn).astype(np.float32, order='C')
#                 dx = np.empty((layer.b, layer.h, layer.w, layer.c), dtype=np.float32, order='C')
#                 # deconv_gemm
#                 deconv_gemm_result = conv_gemm.deconv_gemm_nhwc(weights, dy, dx,
#                                                            vpadding=layer.vpadding, hpadding=layer.hpadding,
#                                                            vstride=layer.vstride, hstride=layer.hstride,
#                                                            vdilation=layer.vdilation, hdilation=layer.hdilation)
#                 # gemm + col2im
#                 dy_cols = dy.transpose((1, 0, 2, 3)).reshape(layer.kn, -1)
#                 w_cols = weights.reshape(layer.kn, -1).T
#                 res = np.matmul(w_cols, dy_cols)
#                 mm_col2im_result = col2im_nhwc_cython(res, dy.shape[0], layer.c, layer.h, layer.w,
#                                                  layer.kh, layer.kw, layer.vpadding, layer.hpadding,
#                                                  layer.vstride, layer.hstride, layer.vdilation, layer.hdilation)
#                 if verbose_test():
#                     print("   {:2}      {:9.7f}".format(n,
#                                                         max([abs(x - y) for x, y
#                                                              in
#                                                              zip(deconv_gemm_result.flatten(),
#                                                                  mm_col2im_result.flatten())])))
#                     if n == 9:
#                         print("Flags for last conv_gemm_result output:")
#                         print(deconv_gemm_result.flags)
#                 self.assertTrue(np.allclose(deconv_gemm_result, mm_col2im_result),
#                                 f"Results differ for AlexNet Cifar and ImageNet layers number {n}")


if __name__ == '__main__':
    try:
        ConvGemm()
    except NameError:
        sys.exit(-1)
    unittest.main()
