"""
Unitary tests for CPU using different tensor formats

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.CheckTensorFormatModels

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckTensorFormatModels

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckTensorFormatModels.test_name
"""

import sys
import unittest
import warnings

import numpy as np

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.layer import LayerError
from pydtnn.model import Model
from pydtnn.tests.model_common import ModelCommonTestCase
from pydtnn.tests.common import verbose_test, Params
from pydtnn.utils.tensor import TensorFormat, format_transpose


class ModelTensorTestCase(ModelCommonTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global ModelCommonTestCase
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend tensor format NHWC"
    model2_desc = "using the CPU backend tensor format NCHW"

    @staticmethod
    def get_model2(model_name: str):
        # Tensor format NCHW
        params = Params()
        params.model_name = model_name
        params.tensor_format = TensorFormat.NCHW.upper()
        params_dict = vars(params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        return model2

    @staticmethod
    def nhwc2nchw(x: np.ndarray):
        if len(x.shape) == 4:
            x = format_transpose(x, TensorFormat.NHWC, TensorFormat.NCHW)
        return np.asarray(x, order="C", copy=None)

    @staticmethod
    def copy_weights_and_biases(model1: Model, model2: Model):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for layer1, layer2 in zip(model1.get_all_layers()[1:], model2.get_all_layers()[1:]):
            if isinstance(layer1, Conv2D):
                assert layer1.grouping is layer2.grouping, f"Both conv_2D layer's grouping must be the same (layer1: {layer1.grouping}, layer2:{layer2.grouping})"

                match layer1.grouping:
                    case Conv2D.Grouping.DEPTHWISE:
                        pass  # Both tensor have the same weights' shape.
                    case Conv2D.Grouping.POINTWISE:
                        # NHWC's shape: ci, co
                        # NCHW's shape: co, ci
                        layer2.weights = np.asarray(layer1.weights.T, order="C", copy=True)
                    case Conv2D.Grouping.STANDARD:
                        # NHWC's shape (indexes from NHWC):
                        # ci (0), kh (1), kw (2), co (3)
                        # NCHW's shape (indexes from NHWC):
                        # co (3), ci (0), kh (1), kw (2)
                        layer2.weights = np.asarray(layer1.weights.transpose(3, 0, 1, 2), order="C", copy=True)
                    case _:
                        raise ValueError(f"Layer grouping (\"{layer1.grouping}\") not in ({list(Conv2D.Grouping)})")
            elif layer2.weights is not None:
                layer2.weights = layer1.weights.copy()
            layer2.biases = layer1.biases.copy() if layer1.biases is not None else None

    def do_model2_forward_pass(self, model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 forward pass in NCHW format
        """
        x2 = [self.nhwc2nchw(x1[0])]
        for i, layer in enumerate(model2.layers[1:], 1):
            if verbose_test():
                print(layer)
            if isinstance(layer, Flatten):
                x2.append(layer.forward(x1[i - 1]))
            else:
                x2.append(layer.forward(self.nhwc2nchw(x1[i - 1])))
        return x2

    def do_model2_backward_pass(self, model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 backward pass in NCHW format
        """
        dx2 = [None] * len(model2.layers)
        dx2[-1] = self.nhwc2nchw(dx1[-1])
        for i, layer in reversed(list(enumerate(model2.layers[1:-1], 1))):
            if verbose_test():
                print(layer)
            dx2[i] = layer.backward(self.nhwc2nchw(dx1[i + 1]))
        return dx2

    def compare_forward(self, model1: Model, x1: list[np.ndarray], model2: Model, x2: list[np.ndarray]):
        assert len(x1) == len(x2), "x1 and x2 should have the same length"
        if verbose_test():
            print()
            print(f"Comparing outputs of both models...")
        for i, layer in enumerate(model1.layers[1:], 1):
            # Skip test on layers that behave randomly
            if not isinstance(layer, Dropout):
                rtol, atol = self.get_tolerance(layer)
                self.assertTrue(np.allclose(self.nhwc2nchw(x1[i]), x2[i], rtol=rtol, atol=atol),
                                f"Forward result from layers {layer.canonical_name_with_id} differ"
                                f" (max diff: {self.max_diff(self.nhwc2nchw(x1[i]), x2[i])}, rtol: {rtol}, atol: {atol})")

    def compare_backward(self, model1: Model, dx1, model2: Model, dx2):
        assert len(dx1) == len(dx2), "dx1 and dx2 should have the same length"
        if verbose_test():
            print()
            print(f"Comparing dw of both models...")
        for i, layer in reversed(list(enumerate(model2.layers[1:], 1))):
            if isinstance(layer, (Conv2D, FC)):
                rtol, atol = self.get_tolerance(layer)
                if len(layer.weights.shape) == 4:
                    # layer.dw:np.ndarray
                    if layer.dw.transpose(1, 2, 3, 0).shape == model1.layers[i].dw.shape:
                        allclose = np.allclose(layer.dw.transpose(1, 2, 3, 0), model1.layers[i].dw, rtol=rtol,
                                               atol=atol)
                        self.assertTrue(allclose,
                                        f"Backward dw from layer {layer.canonical_name_with_id} differ"
                                        f" (max diff: {self.max_diff(layer.dw.transpose(1, 2, 3, 0), model1.layers[i].dw)}, rtol: {rtol}, atol: {atol})")
                else:
                    if layer.dw.shape == model1.layers[i].dw.shape:
                        allclose = np.allclose(layer.dw, model1.layers[i].dw, rtol=rtol, atol=atol)
                        self.assertTrue(allclose,
                                        f"Backward dw from layer {layer.canonical_name_with_id} differ"
                                        f" (max diff: {self.max_diff(layer.dw, model1.layers[i].dw)}, rtol: {rtol}, atol: {atol})")
        if verbose_test():
            print()
            print(f"Comparing db of both models...")
        for i, layer in reversed(list(enumerate(model2.layers[1:], 1))):
            if isinstance(layer, (Conv2D, FC)) and layer.use_bias:
                rtol, atol = self.get_tolerance(layer)
                # layer.db:np.ndarray
                allclose = np.allclose(layer.db, model1.layers[i].db, rtol=rtol, atol=atol)
                self.assertTrue(allclose,
                                f"Backward db from layer {layer.canonical_name_with_id} differ"
                                f" (max diff: {self.max_diff(layer.db, model1.layers[i].db)}, rtol: {rtol}, atol: {atol})")
        if verbose_test():
            print()
            print(f"Comparing dx of both models...")
        for i, layer in reversed(list(enumerate(model2.layers[2:], 2))):
            # Skip test on layers that behave randomly
            if not isinstance(layer, (Dropout, Flatten)):
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
