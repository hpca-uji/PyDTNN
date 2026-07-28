"""Test suite for the BatchNormalizationRelu fused layer implementation."""

import logging
import unittest
from copy import deepcopy

import numpy as np

from pydtnn.activations.relu import Relu
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.input import Input
from pydtnn.model import Model
from pydtnn.tests.abstract.base import D, Params
from pydtnn.tests.abstract.conv_2d import Conv2DTestCase
from pydtnn.utils.tensor import TensorFormat

__all__ = ("BatchNormalizationReluTestCase",)

logger = logging.getLogger(__name__)


class BatchNormalizationReluTestCase(Conv2DTestCase):
    """Tests that BatchNormalization+Relu leads to the same results than BatchNormalizationRelu"""

    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DTestCase
    del Conv2DTestCase

    @staticmethod
    def _get_layers(d: D, deconv: bool = False, trans: bool = False) -> tuple:
        """
        Constructs and initializes a standard layer chain and a fused layer for comparison.

        Args:
            d: Data configuration object.
            deconv: Boolean flag for deconvolution.
            trans: Boolean flag for transposition.

        Returns:
            A tuple containing the standard ConcatenationBlock and the fused BatchNormalizationRelu layer.
        """
        params_chain = Params()
        params_chain.tensor_format = TensorFormat.NCHW
        params_chain.batch_size = d.b
        params_chain.backend = "cpu"  # cpu;conv_2d:gemm
        model_chain = Model(**vars(params_chain))
        model_chain.mode = Model.Mode.EVALUATE
        model_chain.add(Input(model_chain.encode_shape((d.c, d.h, d.w))))
        bn_chain = BatchNormalization()
        relu_chain = Relu()
        chain = ConcatenationBlock([bn_chain, relu_chain])
        model_chain.add(chain)

        params_fuse = deepcopy(params_chain)
        params_fuse.fused_bn_relu = True
        model_fuse = Model(**vars(params_fuse))
        model_fuse.mode = Model.Mode.EVALUATE
        model_fuse.add(Input(model_fuse.encode_shape((d.c, d.h, d.w))))
        bn_fuse = BatchNormalization()
        relu_fuse = Relu()
        model_fuse.add_layers([bn_fuse, relu_fuse])

        model_chain._model_init()
        model_fuse._model_init()
        fuse = model_fuse.layers[1]

        # Set the same initial weights and biases to both layers
        fuse.running_mean = bn_chain.running_mean.copy()
        fuse.running_var = bn_chain.running_var.copy()

        return chain, fuse

    @staticmethod
    def _set_state(layer: BatchNormalization, weights: np.ndarray) -> None:
        """
        Placeholder for setting the state of a BatchNormalization layer.

        Args:
            layer: The BatchNormalization layer instance.
            weights: The weights to be applied.
        """
        pass

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride3(self) -> None:
        """Tests forward and backward pass with specific stride configuration."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride2(self) -> None:
        """Tests forward and backward pass with stride 2."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_handmade_array(self) -> None:
        """Tests forward and backward pass with standard handmade array."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3_filter1x2(self) -> None:
        """Tests forward and backward pass with stride 3 and 1x2 filter."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3_filter1x1(self) -> None:
        """Tests forward and backward pass with stride 3 and 1x1 filter."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_defaults(self) -> None:
        """Tests forward and backward pass with default parameters."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_handmade_array_stride2(self) -> None:
        """Tests forward and backward pass with handmade array and stride 2."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3(self) -> None:
        """Tests forward and backward pass with larger array and stride 3."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride12(self) -> None:
        """Tests forward and backward pass with larger array and stride 12."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_alexnet_imagenet_first_conv2d(self) -> None:
        """Tests forward and backward pass using AlexNet ImageNet configuration."""
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_alexnet_cifar10_first_conv2d(self) -> None:
        """Tests forward and backward pass using AlexNet CIFAR10 configuration."""
        raise NotImplementedError()
