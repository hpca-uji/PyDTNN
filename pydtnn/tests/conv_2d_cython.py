"""Test suite for verifying Conv2D Cython implementation consistency."""

import logging
from copy import deepcopy

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.input import Input
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.initializers import glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DCythonTestCase",)

logger = logging.getLogger(__name__)


class Conv2DCythonTestCase(Conv2DCommonTestCase):
    """Tests that Conv2D with cython leads to the same results than Conv2d with mm and i2c.T"""

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv: bool = False, trans: bool = False) -> tuple[Conv2D, Conv2D]:
        """
        Initializes and returns two Conv2D layers with identical weights.

        One using im2col and the other using Cython backend.
        """
        params_np = Params()
        params_np.tensor_format = TensorFormat.NHWC
        params_np.batch_size = d.b
        params_np.backend = "numpy"
        model_np = Model(**vars(params_np))
        model_np.mode = Model.Mode.TRAIN
        model_np.add(Input(model_np.encode_shape((d.c, d.h, d.w))))
        conv2d_np = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        model_np.add(conv2d_np)

        params_cy = deepcopy(params_np)
        params_cy.backend = "numpy,cython"
        model_cy = Model(**vars(params_cy))
        model_cy.mode = Model.Mode.TRAIN
        model_cy.add(Input(model_cy.encode_shape((d.c, d.h, d.w))))
        conv2d_cy = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        model_cy.add(conv2d_cy)

        model_np._model_init()
        model_cy._model_init()

        # Set the same initial weights and biases to both layers
        conv2d_cy.weights = conv2d_np.weights.copy()
        conv2d_cy.biases = conv2d_np.biases.copy()
        return conv2d_np, conv2d_cy
