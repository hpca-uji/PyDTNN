"""
Unitary tests for GPU with different models' layers

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.GPUModelsTestCase

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.GPUModelsTestCase

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.GPUModelsTestCase.test_name
"""

import sys
import unittest
import warnings
import numpy as np

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.model import Model
from pydtnn.tests import ConvGemmModelsTestCase
from pydtnn.tests.common import verbose_test


class Params:
    pass


class TensorFormatModelsTestCase(ConvGemmModelsTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend tensor format NHWC"
    model2_desc = "using the CPU backend tensor format NCHW"

    rtol_dict = {
        "AdditionBlock": 1e-2,
        "ConcatenationBlock": 1e-2,
    }

    atol_dict = {
        "AdditionBlock": 1e-2,
        "BatchNormalization": 6e-5,
        "ConcatenationBlock": 1e-2,
        "Conv2D": 4e-3,
        "FC": 1e-5,
    }

    @staticmethod
    def get_model2(model_name):
        # Tensor format NCHW
        params = Params()
        params.model_name = model_name
        params.tensor_format = "NCHW"
        return Model(**vars(params))

    @staticmethod
    def nhwc2nchw(x):
        return x.transpose(0, 3, 1, 2) if len(x.shape) == 4 else x

    @staticmethod
    def copy_weights_and_biases(model1, model2):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for layer1, layer2 in zip(model1.get_all_layers()[1:], model2.get_all_layers()[1:]):
            if layer1.canonical_name == "Conv2D":
                layer2.weights = layer1.weights.transpose(3, 0, 1, 2).copy()
            else:
                layer2.weights = layer1.weights.copy()
            layer2.biases = layer1.biases.copy()

    def do_model2_forward_pass(self, model2, x1):
        """
        Model 2 forward pass in NCHW format
        """
        x2 = [self.nhwc2nchw(x1[0])]
        for i, layer in enumerate(model2.layers[1:], 1):
            if verbose_test():
                print(layer)
            if layer.canonical_name == "Flatten":
                x2.append(layer.forward(x1[i - 1]))
            else:
                x2.append(layer.forward(self.nhwc2nchw(x1[i - 1])))
        return x2

    def do_model2_backward_pass(self, model2, dx1):
        """
        Model 2 backward pass in NCHW format
        """
        dx2 = [None] * len(model2.layers)
        dx2[-1] = self.nhwc2nchw(dx1[-1])
        for i, layer in reversed(list(enumerate(model2.layers[2:-1], 2))):
            if verbose_test():
                print(layer)
            dx2[i] = layer.backward(self.nhwc2nchw(dx1[i + 1]))
        return dx2

    def compare_forward(self, model1, x1, model2, x2):
        assert len(x1) == len(x2), "x1 and x2 should have the same length"
        if verbose_test():
            print()
            print(f"Comparing outputs of both models...")
        for i, layer in enumerate(model1.layers[1:], 1):
            # Skip test on layers that behave randomly
            if layer.canonical_name != "Dropout":
                rtol, atol = self.get_tolerance(layer)
                self.assertTrue(np.allclose(self.nhwc2nchw(x1[i]), x2[i], rtol=rtol, atol=atol),
                                f"Forward result from layers {layer.canonical_name_with_id} differ"
                                f" (max diff: {self.max_diff(self.nhwc2nchw(x1[i]), x2[i])}, rtol: {rtol}, atol: {atol})")

    def compare_backward(self, model1, dx1, model2, dx2):
        assert len(dx1) == len(dx2), "dx1 and dx2 should have the same length"
        if verbose_test():
            print()
            print(f"Comparing dx of both models...")
        for i, layer in reversed(list(enumerate(model2.layers[2:], 2))):
            # Skip test on layers that behave randomly
            if layer.canonical_name not in ["Dropout", "Flatten"]:
                rtol, atol = self.get_tolerance(layer)
                if self.nhwc2nchw(dx1[i]).shape == dx2[i].shape:
                    allclose = np.allclose(self.nhwc2nchw(dx1[i]), dx2[i], rtol=rtol, atol=atol)
                else:
                    warnings.warn(f"dx shape on both models for {layer.canonical_name_with_id} differ:"
                                  f" [dx1.shape: {dx1[i].shape}, dx2.shape: {dx2[i].shape}]")
                    # Try flattening both
                    allclose = np.allclose(self.nhwc2nchw(dx1[i]).flatten(), dx2[i].flatten(), rtol=rtol, atol=atol)
                self.assertTrue(allclose,
                                f"Backward result from layer {layer.canonical_name_with_id} differ"
                                f" (max diff: {self.max_diff(self.nhwc2nchw(dx1[i]), dx2[i])}, rtol: {rtol}, atol: {atol})")

if __name__ == '__main__':
    try:
        Model()
    except NameError:
        sys.exit(-1)
    unittest.main()
