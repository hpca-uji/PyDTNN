"""
Test suite for verifying Conv2D GEMM implementation consistency.
"""
import logging
from copy import deepcopy

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.initializers import glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DConvGemmTestCase",)

logger = logging.getLogger(__name__)


class Conv2DConvGemmTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D with conv_gemm leads to the same results than Conv2d with mm and i2c.T
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[Conv2D, Conv2D]:
        """
        Initializes and returns two Conv2D layers with identical weights, one using
        im2col and the other using GEMM backend.
        """
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b
        params.backend = "cpu"
        model_i2c = Model(**vars(params))
        model_i2c.mode = Model.Mode.TRAIN
        params_gc = deepcopy(params)
        params_gc.backend = "cpu;conv_2d:gemm"
        model_cg = Model(**vars(params_gc))
        model_cg.mode = Model.Mode.TRAIN
        conv2d_i2c = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        conv2d_i2c._init_backend_with_model(model_i2c)
        conv2d_cg = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        conv2d_cg._init_backend_with_model(model_cg)
        for layer in (conv2d_i2c, conv2d_cg):
            layer._model_init(prev_shape=model_i2c.encode_shape((d.c, d.h, d.w)), x=None)
        # Set the same initial weights and biases to both layers
        conv2d_cg.weights = conv2d_i2c.weights.copy()
        conv2d_cg.biases = conv2d_i2c.biases.copy()
        return conv2d_i2c, conv2d_cg