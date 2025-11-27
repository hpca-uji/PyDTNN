"""
Performance tests for Conv2D layer using the convGemm library

For running the tests run:
    python profile_conv2d_conv_gemm.py

To obtain a profile, run:
    python3 -m cProfile -o cprofile_conv2d_conv_gemm.prof profile_tests/profile_conv2d_conv_gemm.py

To graphically inspect the profile, run:
    snakeviz cprofile_conv2d_conv_gemm.prof
"""

import cProfile
import time
from copy import deepcopy

import numpy as np

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.utils import random


class D:
    def __init__(self):
        """Default parameters"""
        self.b = 1  # Batch size
        self.c = 1  # Channels per layer
        self.h = 128  # Layers height
        self.w = 100  # Layers width
        self.kn = 1  # Number of filters
        self.kh = 16  # Filters weights height
        self.kw = 10  # Filters weights width
        self.vpadding = 1  # Vertical padding
        self.hpadding = 1  # Horizontal padding
        self.vstride = 1  # Vertical stride
        self.hstride = 1  # Horizontal stride
        self.vdilation = 1  # Vertical dilation
        self.hdilation = 1  # Horizontal dilation

    @property
    def ho(self):
        return (self.h + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1

    @property
    def wo(self):
        return (self.w + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
  dilation        = {self.vdilation} {self.hdilation}
"""


def _print_with_header(header, to_be_printed=None):
    print()
    print("-" * (len(header) + 2))
    print(" {}".format(header))
    print("-" * (len(header) + 2))
    if to_be_printed is not None:
        print(to_be_printed)


class Params:
    pass


def get_conv2d_layers(d: D) -> tuple[Conv2D, Conv2D]:
    params = Params()
    params.batch_size = d.b
    params.conv_variant = "i2c"
    model_i2c = Model(**vars(params))
    params_gc = deepcopy(params)
    params_gc.conv_variant = "gemm"
    model_cg = Model(**vars(params_gc))
    conv2d_i2c = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride), dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer="glorot_uniform", biases_initializer="zeros")
    conv2d_i2c.init_backend_from_model(model_i2c)
    conv2d_i2c.debug = True
    conv2d_cg = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                       padding=(d.vpadding, d.hpadding),
                       stride=(d.vstride, d.hstride), dilation=(d.vdilation, d.hdilation),
                       use_bias=True, weights_initializer="glorot_uniform", biases_initializer="zeros")
    conv2d_cg.init_backend_from_model(model_cg)
    conv2d_cg.debug = True
    for layer in (conv2d_i2c, conv2d_cg):
        layer.initialize(prev_shape=(d.c, d.h, d.w))
    # Set the same initial weights and biases to both layers
    conv2d_cg.weights = conv2d_i2c.weights.copy()
    conv2d_cg.biases = conv2d_i2c.biases.copy()
    return conv2d_i2c, conv2d_cg


class PerfTestConv2DConvGemm:
    """
    Performance test for Conv2D that compares Conv2d with mm and i2c.T
    """

    def _test_forward_backward(self, d, x, weights, print_times=False):
        conv2d_i2c, conv2d_cg = get_conv2d_layers(d)
        conv2d_i2c.weights = weights.copy()
        conv2d_cg.weights = weights.copy()

        # Forward i2c
        print("Forward I2C")
        y_i2c = conv2d_i2c.forward(x)  # First run
        profiler = cProfile.Profile()
        profiler.enable()
        y_i2c = conv2d_i2c.forward(x)
        profiler.disable()
        profiler.print_stats()
        profiler.dump_stats("forward_i2c.prof")
        print()

        # Forward cg
        print("Forward CG")
        y_i2c = conv2d_cg.forward(x)  # First run
        profiler = cProfile.Profile()
        profiler.enable()
        y_i2c = conv2d_cg.forward(x)
        profiler.disable()
        profiler.print_stats()
        profiler.dump_stats("forward_cg.prof")
        print()

        dy = random.random((d.b, d.kn, d.ho, d.wo)).astype(np.float32, order='C')

        # Backward i2c
        print("Backward I2C")
        dx_cg = conv2d_i2c.backward(dy)  # First run
        profiler = cProfile.Profile()
        profiler.enable()
        dx_cg = conv2d_i2c.backward(dy)
        profiler.disable()
        profiler.print_stats()
        profiler.dump_stats("backward_i2c.prof")

        # Backward cg with fallback
        print("Backward CG (with fallback)")
        dx_cg = conv2d_cg.backward(dy)  # First run
        profiler = cProfile.Profile()
        profiler.enable()
        dx_cg = conv2d_cg.backward(dy)
        profiler.disable()
        profiler.print_stats()
        profiler.dump_stats("backward_cg_fallback.prof")

        # Backward cg without fallback
        print("Backward CG (without fallback")
        cg_fallback_to_im2col = conv2d_cg.cg_fallback_to_im2col
        conv2d_cg.cg_fallback_to_im2col = False
        dx_cg = conv2d_cg.backward(dy)  # First run
        profiler = cProfile.Profile()
        profiler.enable()
        dx_cg = conv2d_cg.backward(dy)
        profiler.disable()
        profiler.print_stats()
        profiler.dump_stats("backward_cg_no_fallback.prof")
        conv2d_cg.cg_fallback_to_im2col = cg_fallback_to_im2col

        # cg forward
        tic = time.perf_counter()
        y_cg = conv2d_cg.forward(x)
        toc = time.perf_counter()
        print(f"cg forward: {toc - tic:0.4f} s")
        dy = random.random((d.b, d.kn, d.ho, d.wo)).astype(np.float32, order='C')
        # i2c backward
        tic = time.perf_counter()
        dx_i2c = conv2d_i2c.backward(dy)
        toc = time.perf_counter()
        print("---")
        print(f"i2c backward: {toc - tic:0.4f} s")
        print("")
        # cg backward
        tic = time.perf_counter()
        dx_cg = conv2d_cg.backward(dy)
        toc = time.perf_counter()
        print(f"cg backward: {toc - tic:0.4f} s")
        # All close?
        print("dw all close:", np.allclose(conv2d_i2c.dw, conv2d_cg.dw))
        print("dx all close:", np.allclose(dx_i2c, dx_cg))
        print("Y all close:", np.allclose(y_i2c, y_cg, rtol=1e-5, atol=1e-6))

    def test_forward_backward_alexnet_cifar10_first_conv2d(self):
        """Tests that the AlexNet Cifar10 first Conv2d lead to the same solution on i2c and on conv_gemm"""
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

    def test_forward_backward_alexnet_cifar10_second_conv2d(self):
        """Tests that the AlexNet Cifar10 second Conv2d lead to the same solution on i2c and on conv_gemm"""
        d = D()
        d.b = 64
        d.kn, d.kh, d.kw = (192, 3, 3)
        d.c, d.h, d.w = (64, 8, 8)
        d.vpadding, d.hpadding = (1, 1)
        d.vstride, d.hstride = (1, 1)
        d.vdilation, d.hdilation = (1, 1)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        weights = random.random((d.kn, d.c, d.kh, d.kw)).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights, print_times=True)

    def test_forward_backward_alexnet_cifar10_third_conv2d(self):
        """Tests that the AlexNet Cifar10 third Conv2d lead to the same solution on i2c and on conv_gemm"""
        d = D()
        d.b = 64
        d.kn, d.kh, d.kw = (192, 3, 3)
        d.c, d.h, d.w = (384, 3, 3)
        d.vpadding, d.hpadding = (1, 1)
        d.vstride, d.hstride = (1, 1)
        d.vdilation, d.hdilation = (1, 1)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        weights = random.random((d.kn, d.c, d.kh, d.kw)).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights, print_times=True)

    def test_forward_backward_alexnet_imagenet_first_conv2d(self):
        """Tests that the AlexNet Imagenet first Conv2d lead to the same solution on i2c and on conv_gemm"""
        # id;height;width;channels;kernel_height;kernel_width;kernel_num;stride;padding
        # 2;227;227;3;11;11;96;4;0
        d = D()
        d.b = 64
        d.kn, d.kh, d.kw = (96, 11, 11)
        d.c, d.h, d.w = (3, 227, 227)
        d.vpadding, d.hpadding = (0, 0)
        d.vstride, d.hstride = (4, 4)
        d.vdilation, d.hdilation = (1, 1)
        x = random.random((d.b, d.c, d.h, d.w)).astype(np.float32, order='C')
        weights = random.random((d.kn, d.c, d.kh, d.kw)).astype(np.float32, order='C')
        self._test_forward_backward(d, x, weights, print_times=True)


if __name__ == "__main__":
    c = PerfTestConv2DConvGemm()
    for i in range(1, 2):
        print()
        print("------------")
        print("Iteration:", i)
        print("------------")
        print()
        print("Alexnet for Cifar10")
        print("===================")
        c.test_forward_backward_alexnet_cifar10_first_conv2d()
        # print()
        # print("Alexnet for Imagenet")
        # print("====================")
        # c.test_forward_backward_alexnet_imagenet_first_conv2d()
        # print()
        # print("Alexnet for Cifar10 2n layer")
        # print("============================")
        # c.test_forward_backward_alexnet_cifar10_second_conv2d()
        # print()
        # print("Alexnet for Cifar10 3rd layer")
        # print("=============================")
        # c.test_forward_backward_alexnet_cifar10_third_conv2d()
