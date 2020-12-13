"""
Unitary tests for Conv2D layer using the convGemm library

For running all the tests quietly, execute from the parent directory:
    python -m unittest unittests.TestConv2DConvGemm

For running all the tests verbosely, execute from the parent directory:
    python -m unittest -v unittests.TestConv2DConvGemm

For running an individual test verbosely, execute from the parent directory:
    python -m unittest -v unittests.TestConv2DConvGemm.test_name
"""

import inspect
import math
import sys
import unittest

import numpy as np
from copy import deepcopy

import NN_util
from .tools import Spinner

try:
    from NN_layer import Conv2D
    from NN_model import Model
    from NN_conv_gemm import ConvGemm
    from NN_im2col_cython import im2col_cython
except ModuleNotFoundError:
    print("Please, execute as 'python -m unittest unittests.TestConvGemm'")


def verbose():
    """Returns True if unittest has been called with -v or --verbose options."""
    return '-v' in sys.argv or '--verbose' in sys.argv


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

    @property
    def ho(self):
        return math.floor((self.h + 2 * self.vpadding - self.kh) / self.vstride + 1)

    @property
    def wo(self):
        return math.floor((self.w + 2 * self.hpadding - self.kw) / self.hstride + 1)

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
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


def get_conv2d_layers(d):
    params = Params()
    params.batch_size = 64
    params.enable_conv_gemm = False
    params.cpu_speed = 4000000000000.0
    params.memory_bw = 50000000000.0
    params.network_bw = 1000000000.0
    params.network_lat = 5e-07
    model_i2c = Model(params)
    params_gc = deepcopy(params)
    params_gc.enable_conv_gemm = True
    model_cg = Model(params_gc)
    conv2d_i2c = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        padding=(d.vpadding, d.hpadding), stride=(d.vstride, d.hstride),
                        use_bias=True, weights_initializer="glorot_uniform", biases_initializer="zeros")
    conv2d_i2c.model = model_i2c
    conv2d_cg = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                       padding=(d.vpadding, d.hpadding), stride=(d.vstride, d.hstride),
                       use_bias=True, weights_initializer="glorot_uniform", biases_initializer="zeros")
    conv2d_cg.model = model_cg
    for layer in (conv2d_i2c, conv2d_cg):
        layer.dtype = np.float32
        layer.batch_size = model_i2c.params.batch_size  # batch_size is the same in both models
        layer.tracer = model_i2c.tracer  # tracer is the same on both models
        layer.matmul = getattr(NN_util, "matmul")
        layer.initialize(prev_shape=(d.c, d.h, d.w))
    # Set the same initial weights and biases to both layers
    conv2d_cg.weights = conv2d_i2c.weights.copy()
    conv2d_cg.biases = conv2d_i2c.biases.copy()
    return conv2d_i2c, conv2d_cg


class TestConv2DConvGemm(unittest.TestCase):

    def test_forward_defaults(self):
        """
        Test that the default parameters lead to the same solution on the forward step
        """
        d = D()
        conv2d_i2c, conv2d_cg = get_conv2d_layers(d)
        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32, order='C')
        y_i2c = conv2d_i2c.forward(x)
        y_cg = conv2d_cg.forward(x)
        if verbose():
            _print_with_header("test forward defaults")
            print(y_i2c)
            print(y_cg)
            print("y_i2c.shape:", y_i2c.shape)
            print("y_cg.shape: ", y_cg.shape)
        self.assertTrue(np.allclose(y_i2c, y_cg, rtol=1e-5, atol=1e-6))

    def test_backward_defaults(self):
        """
        Test that the default parameters lead to the same solution on the backward step
        """
        d = D()
        conv2d_i2c, conv2d_cg = get_conv2d_layers(d)
        x = np.random.rand(d.b, d.c, d.h, d.w).astype(np.float32, order='C')
        # Forward pass is required as some intermediate results are stored on it
        conv2d_i2c.forward(x)
        conv2d_cg.forward(x)
        dy = np.random.rand(d.b, d.kn, d.ho, d.wo).astype(np.float32, order='C')
        dx_i2c = conv2d_i2c.backward(dy)
        dx_cg = conv2d_cg.backward(dy)
        dw_allclose = np.allclose(conv2d_i2c.dw, conv2d_cg.dw)
        dx_allclose = np.allclose(dx_i2c, dx_cg)
        if verbose():
            _print_with_header("test backward defaults")
            print(d)
            print("dw_i2c.shape:", conv2d_i2c.dw.shape)
            print("dw_cg.shape: ", conv2d_cg.dw.shape)
            print("dw allclose: ", dw_allclose)
            print("dx_i2c.shape:", dx_i2c.shape)
            print("dx_cg.shape: ", dx_cg.shape)
            print("dx allclose: ", dx_allclose)
        self.assertTrue(dw_allclose, "dw matrices differ")
        self.assertTrue(dx_allclose, "dx return matrices differ")

    def test_handmade_array(self):
        """Tests that manual matrices lead to the same solution"""
        x = np.array([[[[1, 2, 4, 8],
                        [16, 32, 64, 128]]]]).astype(np.float32, order='C')
        weights = np.array([[[[1, 1],
                              [1, 1]]]]).astype(np.float32, order='C')
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (2, 4)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 1
        conv2d_i2c, conv2d_cg = get_conv2d_layers(d)
        conv2d_i2c.weights = weights.copy()
        conv2d_cg.weights = weights.copy()
        # Forward pass is required as some intermediate results are stored on it
        conv2d_i2c.forward(x)
        conv2d_cg.forward(x)
        dy = np.ones((d.b, d.kn, d.ho, d.wo)).astype(np.float32, order='C')
        dx_i2c = conv2d_i2c.backward(dy)
        dx_cg = conv2d_cg.backward(dy)
        dw_allclose = np.allclose(conv2d_i2c.dw, conv2d_cg.dw)
        dx_allclose = np.allclose(dx_i2c, dx_cg)
        if verbose():
            _print_with_header("test backward handmade array")
            print(d)
            print("dw_i2c.shape:", conv2d_i2c.dw.shape)
            print("dw_cg.shape: ", conv2d_cg.dw.shape)
            print("dw allclose: ", dw_allclose)
            print("\ndy_cols:\n", dy.transpose((1, 0, 2, 3)).reshape(d.kn, -1))
            print("x_cols.T:\n", conv2d_i2c.x_cols.T)
            print("dw:\n", conv2d_i2c.dw)
            print("\ndy.transpose:\n", dy.transpose((1, 0, 2, 3)))
            print("cg_x.transpose:\n", conv2d_cg.cg_x.transpose((1, 0, 2, 3)))
            print("dw:\n", conv2d_cg.dw)
            print("dx_i2c.shape:", dx_i2c.shape)
            print("dx_cg.shape: ", dx_cg.shape)
            print("dx allclose: ", dx_allclose)
        self.assertTrue(dw_allclose, "dw matrices differ")
        self.assertTrue(dx_allclose, "dx return matrices differ")

    def test_handmade_array_stride2(self):
        """Tests that manual matrices with stride 2 lead to the same solution"""
        x = np.array([[[[1, 2, 4, 8],
                        [16, 32, 64, 128]]]]).astype(np.float32, order='C')
        weights = np.array([[[[1, 1],
                              [1, 1]]]]).astype(np.float32, order='C')
        d = D()
        d.kn = d.b = d.c = 1
        d.h, d.w = (2, 4)
        d.kh, d.kw = (2, 2)
        d.vpadding = d.hpadding = 0
        d.vstride = d.hstride = 2
        conv2d_i2c, conv2d_cg = get_conv2d_layers(d)
        conv2d_i2c.weights = weights.copy()
        conv2d_cg.weights = weights.copy()
        # Forward pass is required as some intermediate results are stored on it
        conv2d_i2c.forward(x)
        conv2d_cg.forward(x)
        dy = np.ones((d.b, d.kn, d.ho, d.wo)).astype(np.float32, order='C')
        dx_i2c = conv2d_i2c.backward(dy)
        dx_cg = conv2d_cg.backward(dy)
        dw_allclose = np.allclose(conv2d_i2c.dw, conv2d_cg.dw)
        dx_allclose = np.allclose(dx_i2c, dx_cg)
        if verbose():
            _print_with_header("test backward handmade array with stride 2")
            print(d)
            print("dw_i2c.shape:", conv2d_i2c.dw.shape)
            print("dw_cg.shape: ", conv2d_cg.dw.shape)
            print("dw allclose: ", dw_allclose)
            print("\ndy_cols:\n", dy.transpose((1, 0, 2, 3)).reshape(d.kn, -1))
            print("x_cols.T:\n", conv2d_i2c.x_cols.T)
            print("dw:\n", conv2d_i2c.dw)
            print("\ndy.transpose:\n", dy.transpose((1, 0, 2, 3)))
            print("cg_x.transpose:\n", conv2d_cg.cg_x.transpose((1, 0, 2, 3)))
            print("dw:\n", conv2d_cg.dw)
            print("dx_i2c.shape:", dx_i2c.shape)
            print("dx_cg.shape: ", dx_cg.shape)
            print("dx allclose: ", dx_allclose)
        self.assertTrue(dw_allclose, "dw matrices differ")
        self.assertTrue(dx_allclose, "dx return matrices differ")


if __name__ == '__main__':
    try:
        Conv2D()
    except NameError:
        sys.exit(-1)
    unittest.main()
