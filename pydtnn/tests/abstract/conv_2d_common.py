import inspect

import numpy as np

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.tests.abstract.common import verbose_test, D
from pydtnn.tests.abstract.common import TestCase
from pydtnn.utils import print_with_header, random
from pydtnn.utils.tensor import TensorFormat, format_transpose


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
        conv2d_ref, conv2d_test = self._get_layers(d)
        x = conv2d_ref.model.encode_tensor(x).copy()
        if conv2d_ref.model.tensor_format is TensorFormat.NHWC:
            weights = format_transpose(weights, "OIHW", "IHWO").copy()
        self._set_state(conv2d_ref, weights)
        self._set_state(conv2d_test, weights)
        # Forward pass
        y_ref = conv2d_ref.forward(x)
        y_test = conv2d_test.forward(x)
        dy = random.random((d.b, d.kn, d.ho, d.wo)).astype(np.float32, order='C')
        if conv2d_ref.model.tensor_format is TensorFormat.NHWC:
            dy = format_transpose(dy, "NCHW", "NHWC").copy()
        # Backward pass
        dx_ref = conv2d_ref.backward(dy)
        dx_test = conv2d_test.backward(dy)
        # All close?
        dw_allclose = np.allclose(conv2d_ref.dw, conv2d_test.dw)
        dx_allclose = np.allclose(dx_ref, dx_test)
        if verbose_test():
            print_with_header(inspect.stack()[1][3])
            # np.set_printoptions(threshold=50)  # default is 1000
            print(d)
            print("---=[ Forward results ]=---")
            print("y_ref:\n", y_ref)
            print("y_test:\n", y_test)
            print()
            print("---=[ Backward results ]=---")
            print("dx_ref:\n", dx_ref)
            print("dx_test:\n", dx_test)
            print("dw:\n", conv2d_test.dw)
            print("dx allclose: ", dx_allclose)
            print()
            print("---[ dw comparison ]---")
            print("dw_ref.shape:", conv2d_ref.dw.shape)
            print("dw_test.shape: ", conv2d_test.dw.shape)
            print("dw allclose: ", dw_allclose)
            if print_times:
                forward_ref_t = timeit(lambda: conv2d_ref.forward(x), number=10) / 10
                forward_test_t = timeit(lambda: conv2d_test.forward(x), number=10) / 10
                backward_ref_t = timeit(lambda: conv2d_ref.backward(dy), number=10) / 10
                backward_test_t = timeit(lambda: conv2d_test.backward(dy), number=10) / 10
                print()
                print("---[ times comparison ]---")
                print("            ref     test")
                print("         +-------+--------+")
                print(" forward | {:.3f} | {:.3f} |".format(forward_ref_t, forward_test_t))
                print("         +-------+--------+")
                print("backward | {:.3f} | {:.3f} |".format(backward_ref_t, backward_test_t))
                print("         +-------+--------+")
                print("           {:.3f}   {:.3f}  ".format(forward_ref_t + backward_ref_t,
                                                            forward_test_t + backward_test_t))
        # self.assertTrue(np.allclose(y_ref, y_test, rtol=1e-5, atol=1e-6), f"y matrices differ")
        self.assertTrue(np.allclose(y_ref, y_test), "y matrices differ")
        self.assertTrue(dw_allclose, "dw matrices differ")
        self.assertTrue(dx_allclose, "dx return matrices differ")

    def test_forward_defaults(self):
        """
        Test that the default parameters lead to the same solution on the forward step
        """
        d = D()
        conv2d_ref, conv2d_test = self._get_layers(d)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        x = conv2d_ref.model.encode_tensor(x).copy()
        y_ref = conv2d_ref.forward(x)
        y_test = conv2d_test.forward(x)
        if verbose_test():
            print_with_header("test forward defaults")
            print(y_ref)
            print(y_test)
            print("y_ref.shape:", y_ref.shape)
            print("y_test.shape: ", y_test.shape)
        allclose = np.allclose(y_ref, y_test, rtol=1e-5, atol=1e-6)
        self.assertTrue(allclose)

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
