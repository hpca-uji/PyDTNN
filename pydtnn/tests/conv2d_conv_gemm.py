"""
Unitary tests for Conv2D layer using the convGemm library

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.Conv2DConvGemmTestCase

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.Conv2DConvGemmTestCase

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.Conv2DConvGemmTestCase.test_name
"""

import inspect
import sys
import unittest
from copy import deepcopy

import numpy as np

from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.tests.common import verbose_test, D
from pydtnn.tests.tools import print_with_header
from ..model import Model


class Params:
    pass


def get_conv2d_cpu_layers(d, deconv=False, trans=False):
    params = Params()
    params.batch_size = d.b
    params.enable_conv_gemm = False
    params.tensor_format = 'NCHW'
    model_i2c = Model(**vars(params))
    params_gc = deepcopy(params)
    params_gc.enable_conv_gemm = True
    params_gc.conv_gemm_cache = True
    params_gc.conv_gemm_fallback_to_im2col = False
    params_gc.conv_gemm_deconv = deconv
    params_gc.conv_gemm_trans = trans
    model_cg = Model(**vars(params_gc))
    conv2d_i2c = Conv2DCPU(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                           padding=(d.vpadding, d.hpadding), stride=(d.vstride, d.hstride),
                           use_bias=True, weights_initializer="glorot_uniform", biases_initializer="zeros")
    conv2d_i2c.set_model(model_i2c)
    conv2d_cg = Conv2DCPU(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                          padding=(d.vpadding, d.hpadding), stride=(d.vstride, d.hstride),
                          use_bias=True, weights_initializer="glorot_uniform", biases_initializer="zeros")
    conv2d_cg.set_model(model_cg)
    for layer in (conv2d_i2c, conv2d_cg):
        layer.initialize(prev_shape=(d.c, d.h, d.w))
    # Set the same initial weights and biases to both layers
    conv2d_cg.weights = conv2d_i2c.weights.copy()
    conv2d_cg.biases = conv2d_i2c.biases.copy()
    return conv2d_i2c, conv2d_cg


class Conv2DConvGemmTestCase(unittest.TestCase):
    """
    Tests that Conv2D with conv_gemm leads to the same results than Conv2d with mm and i2c.T
    """

    x_2x4 = np.array([[[[1, 2, 4, 8],
                        [16, 32, 64, 128]]]]).astype(np.float32, order='C')

    x_4x4 = np.array([[[[1, 2, 4, 8],
                        [16, 32, 64, 128],
                        [1, 2, 4, 8],
                        [16, 32, 64, 128]]]]).astype(np.float32, order='C')

    x_4x8 = np.array([[[[1, 2, 4, 8, 9, 10, 11, 12],
                        [16, 32, 64, 128, 129, 130, 131, 132],
                        [1, 2, 4, 8, 9, 10, 11, 12],
                        [16, 32, 64, 128, 129, 130, 131, 132]]]]).astype(np.float32, order='C')

    x_8x8 = np.array([[[[11, 12, 13, 14, 15, 16, 17, 18],
                        [21, 22, 23, 24, 25, 26, 27, 28],
                        [31, 32, 33, 34, 35, 36, 37, 38],
                        [41, 42, 43, 44, 45, 46, 47, 48],
                        [51, 52, 53, 54, 55, 56, 57, 58],
                        [61, 62, 63, 64, 65, 66, 67, 68],
                        [71, 72, 73, 74, 75, 76, 77, 78],
                        [81, 82, 83, 84, 85, 86, 87, 88]]]]).astype(np.float32, order='C')

    w_1x1 = np.array([[[[1]]]]).astype(np.float32, order='C')

    w_1x2 = np.array([[[[1, 1]]]]).astype(np.float32, order='C')

    w_2x2 = np.array([[[[1, 1],
                        [1, 1]]]]).astype(np.float32, order='C')

    w_3x3 = np.array([[[[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]]]).astype(np.float32, order='C')

    def _test_forward_backward(self, d, x, weights, print_times=False):
        self._test_forward_backward_inner(d, x, weights, print_times=print_times, deconv=False, trans=False)
        self._test_forward_backward_inner(d, x, weights, print_times=print_times, deconv=True, trans=False)
        self._test_forward_backward_inner(d, x, weights, print_times=print_times, deconv=False, trans=True)
        self._test_forward_backward_inner(d, x, weights, print_times=print_times, deconv=True, trans=True)

    def _test_forward_backward_inner(self, d, x, weights, print_times=False, deconv=False, trans=False):
        from timeit import timeit
        conv2d_i2c, conv2d_cg = get_conv2d_cpu_layers(d, deconv, trans)
        conv2d_i2c.weights = weights.copy()
        conv2d_cg.weights = weights.copy()
        # Forward pass
        y_i2c = conv2d_i2c.forward(x)
        y_cg = conv2d_cg.forward(x)
        dy = np.random.rand(d.b, d.kn, d.ho, d.wo).astype(np.float32, order='C')
        # Backward pass
        dx_i2c = conv2d_i2c.backward(dy)
        dx_cg = conv2d_cg.backward(dy)
        # All close?
        dw_allclose = np.allclose(conv2d_i2c.dw, conv2d_cg.dw)
        dx_allclose = np.allclose(dx_i2c, dx_cg)
        if verbose_test():
            print_with_header(inspect.stack()[1][3])
            # np.set_printoptions(threshold=50)  # default is 1000
            print(d)
            print("---=[ Forward results ]=---")
            print("y_i2c:\n", y_i2c)
            print("y_cg:\n", y_cg)
            print()
            print("---=[ dy_cols * i2c.T ]=---")
            print("dy_cols:\n", dy.transpose((1, 0, 2, 3)).reshape(d.kn, -1))
            print("x_cols.T:\n", conv2d_i2c.x_cols.T)
            print("dw:\n", conv2d_i2c.dw)
            print()
            print("---=[ conv_gemm(dy * x indexed) ]=---")
            print("dy:\n", dy.transpose((1, 0, 2, 3)))
            try:
                print("x:\n", conv2d_cg.cg_x.transpose((1, 0, 2, 3)))
            except AttributeError:
                pass
            try:
                print("x indexed:\n", conv2d_cg.cg_x_indexed)
            except AttributeError:
                pass
            print("dw:\n", conv2d_cg.dw)
            print()
            print("---[ dw comparison ]---")
            print("dw_i2c.shape:", conv2d_i2c.dw.shape)
            print("dw_cg.shape: ", conv2d_cg.dw.shape)
            print("dw allclose: ", dw_allclose)
            print()
            print("---[ dx comparison ]---")
            print("dx_i2c.shape:", dx_i2c.shape)
            if dx_i2c.size < 30:
                print(dx_i2c)
            print("dx_cg.shape: ", dx_cg.shape)
            if dx_cg.size < 30:
                print(dx_cg)
            print("dx allclose: ", dx_allclose)
            if print_times:
                forward_i2c_t = timeit(lambda: conv2d_i2c.forward(x), number=10) / 10
                forward_cg_t = timeit(lambda: conv2d_cg.forward(x), number=10) / 10
                backward_i2c_t = timeit(lambda: conv2d_i2c.backward(dy), number=10) / 10
                backward_cg_t = timeit(lambda: conv2d_cg.backward(dy), number=10) / 10
                print()
                print("---[ times comparison ]---")
                print("            i2c     cg")
                print("         +-------+--------+")
                print(" forward | {:.3f} | {:.3f} |".format(forward_i2c_t, forward_cg_t))
                print("         +-------+--------+")
                print("backward | {:.3f} | {:.3f} |".format(backward_i2c_t, backward_cg_t))
                print("         +-------+--------+")
                print("           {:.3f}   {:.3f}  ".format(forward_i2c_t + backward_i2c_t,
                                                            forward_cg_t + backward_cg_t))
        # self.assertTrue(np.allclose(y_i2c, y_cg, rtol=1e-5, atol=1e-6),
        #                 f"y matrices differ (deconv={deconv}, trans={trans})")
        self.assertTrue(np.allclose(y_i2c, y_cg), f"y matrices differ (deconv={deconv}, trans={trans})")
        self.assertTrue(dw_allclose, f"dw matrices differ (deconv={deconv}, trans={trans})")
        self.assertTrue(dx_allclose, f"dx return matrices differ (deconv={deconv}, trans={trans})")

    def test_forward_defaults(self):
        """
        Test that the default parameters lead to the same solution on the forward step
        """
        d = D()
        conv2d_i2c, conv2d_cg = get_conv2d_cpu_layers(d)
        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32, order='C')
        y_i2c = conv2d_i2c.forward(x)
        y_cg = conv2d_cg.forward(x)
        if verbose_test():
            print_with_header("test forward defaults")
            print(y_i2c)
            print(y_cg)
            print("y_i2c.shape:", y_i2c.shape)
            print("y_cg.shape: ", y_cg.shape)
        self.assertTrue(np.allclose(y_i2c, y_cg, rtol=1e-5, atol=1e-6))

    def test_forward_backward_defaults(self):
        """
        Test that the default parameters lead to the same solution on the backward step
        """
        d = D()
        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32, order='C')
        weights = np.random.rand(d.kn, d.c, d.kh, d.kw).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_handmade_array(self):
        """Tests that manual matrices lead to the same solution"""
        x = self.x_2x4
        weights = self.w_2x2
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (2, 4)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 1
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_handmade_array_stride2(self):
        """Tests that manual matrices with stride 2 lead to the same solution"""
        x = self.x_2x4
        weights = self.w_2x2
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (2, 4)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 2
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_larger_handmade_array_stride2(self):
        """Tests that larger manual matrices with stride 2 lead to the same solution"""
        x = self.x_4x4
        weights = self.w_2x2
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (4, 4)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 2
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_larger_handmade_array_stride3(self):
        """Tests that larger manual matrices with stride 3 lead to the same solution"""
        x = self.x_4x4
        weights = self.w_2x2
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (4, 4)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 3
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_even_larger_handmade_array_stride3(self):
        """Tests that even larger manual matrices with stride 3 lead to the same solution on i2c and on conv_gemm"""
        x = self.x_4x8
        weights = self.w_2x2
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (4, 8)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 3
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_even_larger_handmade_array_stride3_filter1x2(self):
        """Tests that even larger manual matrices with stride 3 lead to the same solution on i2c and on conv_gemm"""
        x = self.x_4x8
        weights = self.w_1x2
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (4, 8)
        d.kh, d.kw = (1, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 3
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_even_larger_handmade_array_stride3_filter1x1(self):
        """Tests that even larger manual matrices with stride 3 lead to the same solution on i2c and on conv_gemm"""
        x = self.x_4x8
        weights = self.w_1x1
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (4, 8)
        d.kh, d.kw = (1, 1)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 3
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_even_larger_handmade_array_stride12(self):
        """Tests that even larger manual matrices with strides 1, 2 lead to the same solution on i2c and on conv_gemm"""
        x = self.x_8x8
        weights = self.w_3x3
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (8, 8)
        d.kh, d.kw = (3, 3)
        d.vpadding = d.hpadding = 0
        d.vstride = 1
        d.hstride = 2
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_alexnet_cifar10_first_conv2d(self):
        """Tests that the AlexNet cifar10 first Conv2d lead to the same solution on i2c and on conv_gemm"""
        d = D()
        d.b = 64
        d.kn, d.kh, d.kw = (64, 3, 3)
        d.c, d.h, d.w = (3, 32, 32)
        d.vpadding, d.hpadding = (1, 1)
        d.vstride, d.hstride = (2, 2)
        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32, order='C')
        weights = np.random.rand(d.kn, d.c, d.kh, d.kw).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights, print_times=True)

    def test_forward_backward_alexnet_imagenet_first_conv2d(self):
        """Tests that the AlexNet ImageNet first Conv2d lead to the same solution on i2c and on conv_gemm"""
        # id;height;width;channels;kernel_height;kernel_width;kernel_num;stride;padding
        # 2;227;227;3;11;11;96;4;0
        d = D()
        d.b = 64
        d.kn, d.kh, d.kw = (96, 11, 11)
        d.c, d.h, d.w = (3, 227, 227)
        d.vpadding, d.hpadding = (1, 1)
        d.vstride, d.hstride = (4, 4)
        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32, order='C')
        weights = np.random.rand(d.kn, d.c, d.kh, d.kw).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights, print_times=True)


if __name__ == '__main__':
    try:
        Conv2DCPU()
    except NameError:
        sys.exit(-1)
    unittest.main()
