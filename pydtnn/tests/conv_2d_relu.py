"""
Test suite for verifying the correctness of the fused Conv2D+ReLU layer implementation.
"""
import logging
import unittest

from pydtnn.activations.relu import Relu
from pydtnn.backends.fuse.layers.conv_2d_relu import Conv2DRelu
from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.initializers import glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat

__all__ = ("Conv2DReluTestCase",)

logger = logging.getLogger(__name__)


# TODO: Mirar esto.


class Conv2DReluTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D+Relu leads to the same results than Conv2DRelu
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[ConcatenationBlock, AbstractConv2D]:
        """
        Constructs and initializes a standard Conv2D+ReLU chain and a fused Conv2DRelu layer.

        Args:
            d: Configuration parameters for the layer dimensions.
            deconv: Boolean flag for deconvolution (unused).
            trans: Boolean flag for transposed convolution (unused).

        Returns:
            A tuple containing the standard ConcatenationBlock and the fused Conv2DRelu layer.
        """
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b  # type: ignore (it's okay)
        params.backend = "cpu;conv_2d:gemm"
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d = Conv2D(
            nfilters=d.kn,
            filter_shape=(d.kh, d.kw),
            padding=(d.vpadding, d.hpadding),
            stride=(d.vstride, d.hstride),
            dilation=(d.vdilation, d.hdilation),
            use_bias=True,
            weights_initializer=glorot_uniform,
            biases_initializer=zeros,
        )

        relu = Relu()
        chain = ConcatenationBlock([conv2d, relu])
        shape = (d.c, d.h, d.w)
        chain._init_backend_with_model(model)
        chain._model_init(prev_shape=shape, x=None)

        from_parent = relu.__dict__ | conv2d.__dict__
        fuse = Conv2DRelu(from_parent=from_parent)
        fuse.init_backend_with_model(model)
        fuse.__dict__.update(from_parent)
        fuse.initialize(prev_shape=shape, x=None)

        # Set the same initial weights and biases to both layers
        fuse.weights = conv2d.weights.copy()
        fuse.biases = conv2d.biases.copy()

        return chain, fuse

    @staticmethod
    def _set_state(layer: Conv2D, weights) -> None:
        """
        Synchronizes weights between the standard layer and the fused layer.

        Args:
            layer: The target layer to update.
            weights: The weight tensor to copy.
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