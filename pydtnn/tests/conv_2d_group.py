"""
Test suite for verifying grouped 2D convolution operations in PyDTNN.
"""

import logging
from copy import deepcopy

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.input import Input
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.initializers import glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DGroupTestCase",)

logger = logging.getLogger(__name__)


class Conv2DGroupTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D with Depth+Pair leads to the same results than Conv2D Standard
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[AbstractConv2DNumpy, AbstractConv2DNumpy]:
        """
        Constructs and initializes standard Conv2D and grouped Conv2D layers for comparison.

        Args:
            d: Data configuration object.
            deconv: Boolean flag for deconvolution (unused).
            trans: Boolean flag for transposed convolution (unused).

        Returns:
            A tuple containing the standard Conv2D layer and the grouped ConcatenationBlock.
        """
        params_chain = Params()
        params_chain.tensor_format = TensorFormat.NHWC.upper()
        params_chain.batch_size = d.b
        model_chain = Model(**vars(params_chain))
        model_chain.mode = Model.Mode.TRAIN
        model_chain.add(Input(model_chain.encode_shape((d.c, d.h, d.w))))
        conv2d_depth = Conv2DDepthwise(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        conv2d_pair = Conv2DPointwise(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        chain = ConcatenationBlock([conv2d_depth, conv2d_pair])
        model_chain.add(chain)

        params_fuse = deepcopy(params_chain)
        model_fuse = Model(**vars(params_fuse))
        model_fuse.mode = Model.Mode.TRAIN
        model_fuse.add(Input(model_fuse.encode_shape((d.c, d.h, d.w))))
        conv2d_fuse = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        model_fuse.add(conv2d_fuse)

        model_chain._model_init()
        model_fuse._model_init()
        fuse = model_fuse.layers[1]

        # Set the same initial weights and biases to both layers
        conv2d_depth.weights = fuse.weights.copy()
        conv2d_depth.biases = fuse.biases.copy()
        conv2d_pair.weights = fuse.weights.copy()
        conv2d_pair.biases = fuse.biases.copy()

        return conv2d_fuse, chain  # type: ignore

    @staticmethod
    def _set_state(layer: Conv2D, weights) -> None:
        """
        Synchronizes weights between standard and grouped layer implementations.

        Args:
            layer: The target layer or block to update.
            weights: The weight tensor to assign.
        """
        if isinstance(layer, ConcatenationBlock):
            layer.paths[0][0].weights = weights.copy()
            layer.paths[0][1].weights = weights.copy()
        else:
            layer.weights = weights.copy()
