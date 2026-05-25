"""
Unit tests for the Conv2D and BatchNormalization fusion layer.
"""

import logging
import unittest
from copy import deepcopy

from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.input import Input
from pydtnn.libs.convGemm import is_conv_gemm_available
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.initializers import glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DBatchNormalizationTestCase",)


logger = logging.getLogger(__name__)

# TODO: MIRAR ESTO.


@unittest.skipUnless(is_conv_gemm_available, "requires ConvGemm")
class Conv2DBatchNormalizationTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D+BatchNormalization leads to the same results than Conv2DBatchNormalization
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[AbstractConv2DNumpy, AbstractConv2DNumpy]:
        """
        Creates and initializes a standard Conv2D+BN chain and a fused Conv2DBatchNormalization layer.

        Args:
            d: Data configuration object.
            deconv: Whether to use deconvolution.
            trans: Whether to use transposed convolution.

        Returns:
            A tuple containing the concatenated layer chain and the fused layer.
        """
        params_chain = Params()
        params_chain.tensor_format = TensorFormat.NCHW.upper()
        params_chain.batch_size = d.b
        params_chain.backend = "cpu;conv_2d:gemm"
        model_chain = Model(**vars(params_chain))
        model_chain.mode = Model.Mode.EVALUATE
        model_chain.add(Input(model_chain.encode_shape((d.c, d.h, d.w))))
        conv2d_chain = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )
        bn_chain = BatchNormalization()
        chain = ConcatenationBlock([conv2d_chain, bn_chain])
        model_chain.add(chain)

        params_fuse = deepcopy(params_chain)
        params_fuse.enable_fused_conv_bn = True
        model_fuse = Model(**vars(params_fuse))
        model_fuse.mode = Model.Mode.EVALUATE
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
        bn_fuse = BatchNormalization()
        model_fuse.add_layers([conv2d_fuse, bn_fuse])

        model_chain._model_init()
        model_fuse._model_init()
        fuse = model_fuse.layers[1]

        # Set the same initial weights and biases to both layers
        fuse.weights = conv2d_chain.weights.copy()
        fuse.biases = conv2d_chain.biases.copy()

        return chain, fuse  # type: ignore

    @staticmethod
    def _set_state(layer: Conv2D, weights) -> None:
        """
        Sets the weights for the provided layer or concatenation block.

        Args:
            layer: The layer or block to update.
            weights: The weights to assign.
        """
        if isinstance(layer, ConcatenationBlock):
            layer.paths[0][0].weights = weights.copy()
        else:
            layer.weights = weights.copy()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride3(self):
        """Tests forward and backward pass with stride 3."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride2(self):
        """Tests forward and backward pass with stride 2."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_handmade_array(self):
        """Tests forward and backward pass with standard handmade array."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3_filter1x2(self):
        """Tests forward and backward pass with stride 3 and 1x2 filter."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3_filter1x1(self):
        """Tests forward and backward pass with stride 3 and 1x1 filter."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_defaults(self):
        """Tests forward and backward pass with default parameters."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_handmade_array_stride2(self):
        """Tests forward and backward pass with handmade array and stride 2."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3(self):
        """Tests forward and backward pass with larger array and stride 3."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride12(self):
        """Tests forward and backward pass with larger array and stride 12."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_alexnet_imagenet_first_conv2d(self):
        """Tests forward and backward pass using AlexNet ImageNet configuration."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_alexnet_cifar10_first_conv2d(self):
        """Tests forward and backward pass using AlexNet CIFAR10 configuration."""
        raise NotImplementedError()
