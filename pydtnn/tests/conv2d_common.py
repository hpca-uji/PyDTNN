import inspect

import numpy as np

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.tests.common import verbose_test, D
from pydtnn.tests.common import TestCase
from pydtnn.utils import print_with_header, random


class Conv2DCommonTestCase[T: Conv2D](TestCase):
    """
    Tests that A layer leads to the same results than B layer
    """

    @staticmethod
    def _get_layers(d: D) -> tuple[T, T]:
        raise NotImplementedError()

    @staticmethod
    def _set_state(layer: Conv2D, weights) -> None:
        layer.weights = weights.copy()

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

    def _test_forward_backward(self, d: D, x: np.ndarray, weights: np.ndarray, print_times=False):
        from timeit import timeit
        conv2d_i2c, conv2d_cg = self._get_layers(d)
        self._set_state(conv2d_i2c, weights)
        self._set_state(conv2d_cg, weights)
        # Forward pass
        y_i2c = conv2d_i2c.forward(x)
        y_cg = conv2d_cg.forward(x)
        dy = random.random((d.b, d.kn, d.ho, d.wo)).astype(np.float32, order='C')
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
        # self.assertTrue(np.allclose(y_i2c, y_cg, rtol=1e-5, atol=1e-6), f"y matrices differ")
        self.assertTrue(np.allclose(y_i2c, y_cg), "y matrices differ")
        self.assertTrue(dw_allclose, "dw matrices differ")
        self.assertTrue(dx_allclose, "dx return matrices differ")

    def test_forward_defaults(self):
        """
        Test that the default parameters lead to the same solution on the forward step
        """
        d = D()
        conv2d_i2c, conv2d_cg = self._get_layers(d)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
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
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        weights = random.random((d.kn, d.c, d.kh, d.kw)).astype(np.float32, order='C')
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
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
        d.vdilation = d.hdilation = 1
        self._test_forward_backward(d, x, weights)

    def test_forward_backward_alexnet_cifar10_first_conv2d(self):
        """Tests that the AlexNet cifar10 first Conv2d lead to the same solution on i2c and on conv_gemm"""
        d = D()
        d.b = 64
        d.kn, d.kh, d.kw = (64, 3, 3)
        d.c, d.h, d.w = (3, 32, 32)
        d.vpadding, d.hpadding = (1, 1)
        d.vstride, d.hstride = (2, 2)
        d.vdilation, d.hdilation = (1, 1)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        weights = random.random((d.kn, d.c, d.kh, d.kw)).astype(np.float32, order='C')
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
        d.vdilation, d.hdilation = (1, 1)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        weights = random.random((d.kn, d.c, d.kh, d.kw)).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights, print_times=True)
