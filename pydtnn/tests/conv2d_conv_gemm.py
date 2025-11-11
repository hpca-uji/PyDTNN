import sys
import unittest
from copy import deepcopy

from pydtnn.model import Model
from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.tests.common import D
from pydtnn.tests.common import Params
from pydtnn.tests.conv2d_common import Conv2DCommonTestCase
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.initializers import glorot_uniform, zeros


class Conv2DConvGemmTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D with conv_gemm leads to the same results than Conv2d with mm and i2c.T
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[Conv2DCPU, Conv2DCPU]:
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b
        params.enable_conv_gemm = False
        params.enable_best_of = False
        model_i2c = Model(**vars(params))
        model_i2c.mode = Model.Mode.TRAIN
        params_gc = deepcopy(params)
        params_gc.enable_conv_gemm = True
        model_cg = Model(**vars(params_gc))
        model_cg.mode = Model.Mode.TRAIN
        conv2d_i2c = Conv2DCPU(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                               padding=(d.vpadding, d.hpadding),
                               stride=(d.vstride, d.hstride),
                               dilation=(d.vdilation, d.hdilation),
                               use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d_i2c.set_model(model_i2c)
        conv2d_cg = Conv2DCPU(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                              padding=(d.vpadding, d.hpadding),
                              stride=(d.vstride, d.hstride),
                              dilation=(d.vdilation, d.hdilation),
                              use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d_cg.set_model(model_cg)
        for layer in (conv2d_i2c, conv2d_cg):
            layer.initialize(prev_shape=(d.c, d.h, d.w))
        # Set the same initial weights and biases to both layers
        conv2d_cg.weights = conv2d_i2c.weights.copy()
        conv2d_cg.biases = conv2d_i2c.biases.copy()
        return conv2d_i2c, conv2d_cg
