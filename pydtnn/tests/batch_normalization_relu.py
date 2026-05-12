"""
Test suite for the BatchNormalizationRelu fused layer implementation.
"""

import logging
import unittest

from pydtnn.activations.relu import Relu
from pydtnn.backends.fuse.layers.batch_normalization_relu import BatchNormalizationRelu
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.tensor import TensorFormat

__all__ = ("BatchNormalizationReluTestCase",)

logger = logging.getLogger(__name__)


# TODO: Mirar esto.


class BatchNormalizationReluTestCase(Conv2DCommonTestCase):
    """
    Tests that BatchNormalization+Relu leads to the same results than BatchNormalizationRelu
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple:
        """
        Constructs and initializes a standard layer chain and a fused layer for comparison.

        Args:
            d: Data configuration object.
            deconv: Boolean flag for deconvolution.
            trans: Boolean flag for transposition.

        Returns:
            A tuple containing the standard ConcatenationBlock and the fused BatchNormalizationRelu layer.
        """
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b
        params.backend = "cpu;conv_2d:gemm"
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        bn = BatchNormalization()
        relu = Relu()
        chain = ConcatenationBlock([bn, relu])
        shape = (d.c, d.h, d.w)
        chain._init_backend_with_model(model)
        chain._model_init(prev_shape=shape, x=None)

        from_parent = relu.__dict__ | bn.__dict__
        fuse = BatchNormalizationRelu(from_parent=from_parent)
        fuse.init_backend_with_model(model)
        fuse.__dict__.update(from_parent)
        fuse.initialize(prev_shape=shape, x=None)

        # Set the same initial weights and biases to both layers
        fuse.running_mean = bn.running_mean.copy()
        fuse.running_var = bn.running_var.copy()

        return chain, fuse

    @staticmethod
    def _set_state(layer: BatchNormalization, weights) -> None:
        """
        Placeholder for setting the state of a BatchNormalization layer.

        Args:
            layer: The BatchNormalization layer instance.
            weights: The weights to be applied.
        """
        pass

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride3(self):
        """Tests forward and backward pass with specific stride configuration."""
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
