"""
Unitary tests for ConvGemm with different models' layers

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.CheckConvGemmModels

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckConvGemmModels

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckConvGemmModels.test_name
"""

import sys
import unittest
import warnings

import numpy as np
from rich.console import Console

from pydtnn import losses
from pydtnn.model import Model, TRAIN_MODE
from pydtnn.tests.common import verbose_test
from pydtnn.tests.tools import print_with_header


class Params:
    pass


class CheckConvGemmModels(unittest.TestCase):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using Im2Col+MM"
    model2_desc = "using ConvGemm"

    rtol_default = 1e-5
    atol_default = 1e-6
    rtol_dict = {
        "AdditionBlock": 1e-4,
    }
    atol_dict = {
        "AdditionBlock": 3e-4,
        "ConcatenationBlock": 5e-4,
        "Conv2D": 1e-5,
    }

    def get_tolerance(self, layer):
        rtol, atol = (self.rtol_dict.get(layer.canonical_name, self.rtol_default),
                      self.atol_dict.get(layer.canonical_name, self.atol_default))
        if layer.canonical_name in ("AdditionBlock", "ConcatenationBlock"):
            rtol *= len(layer.children)
            atol *= len(layer.children)
        return rtol, atol

    @staticmethod
    def get_model1_and_loss_func(model_name, overwrite_params=None):
        # CPU model with no convGemm
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.enable_conv_gemm = False
        params.conv_gemm_cache = False
        params.tensor_format = "NHWC"
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        model1 = Model(**params_dict)
        # loss function
        loss = model1.loss_func
        local_batch_size = model1.batch_size
        loss_func = getattr(losses, loss)(shape=(local_batch_size, *model1.layers[-1].shape), model=model1)
        return model1, loss_func

    @staticmethod
    def get_model2(model_name, overwrite_params=None):
        # CPU model with convGemm
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.enable_conv_gemm = True
        params.conv_gemm_cache = True
        params.conv_gemm_trans = True
        params.conv_gemm_deconv = True
        params.tensor_format = "NHWC"
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        return Model(**params_dict)

    @staticmethod
    def copy_weights_and_biases(model1, model2):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        for layer1, layer2 in zip(model1.get_all_layers()[1:], model2.get_all_layers()[1:]):
            layer2.weights = layer1.weights.copy()
            layer2.biases = layer1.biases.copy()

    @staticmethod
    def get_first_dx(model, loss_func, x):
        # random y target
        y_targ = np.random.rand(*x.shape).astype(np.float32, order='C')
        # obtain first dx1
        global_batch_size = model.batch_size
        loss, dx = loss_func(x, y_targ, global_batch_size)
        return dx

    @staticmethod
    def max_diff(x1, x2):
        return np.max([abs(x1 - x2) for x1, x2 in zip(x1, x2)])

    @staticmethod
    def do_model1_forward_pass(model1):
        """
        Model 1 forward pass
        """
        x1 = [np.random.rand(model1.batch_size, *model1.layers[0].shape).astype(np.float32, order='C'), ]
        # Store results from layer 1 to last layer on Model 1
        for i, layer in enumerate(model1.layers[1:], 1):
            if verbose_test():
                print(layer)
            x1.append(layer.forward(x1[i - 1]))
        return x1

    @staticmethod
    def do_model2_forward_pass(model2, x1):
        """
        Model 2 forward pass
        """
        x2 = [x1[0]]
        for i, layer in enumerate(model2.layers[1:], 1):
            if verbose_test():
                print(layer)
            x2.append(layer.forward(x1[i - 1]))
        return x2

    @staticmethod
    def do_model1_backward_pass(model1, first_dx):
        """
        Model 1 backward pass
        """
        dx1 = [None] * len(model1.layers)
        dx1[-1] = first_dx
        for i, layer in reversed(list(enumerate(model1.layers[1:-1], 1))):
            if verbose_test():
                print(layer)
            dx1[i] = layer.backward(dx1[i + 1])
        return dx1

    @staticmethod
    def do_model2_backward_pass(model2, dx1):
        """
        Model 2 backward pass
        """
        dx2 = [None] * len(model2.layers)
        dx2[-1] = dx1[-1]
        for i, layer in reversed(list(enumerate(model2.layers[1:-1], 1))):
            if verbose_test():
                print(layer)
            dx2[i] = layer.backward(dx1[i + 1])
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
                self.assertTrue(np.allclose(x1[i], x2[i], rtol=rtol, atol=atol),
                                f"Forward result from layers {layer.canonical_name_with_id} differ"
                                f" (max diff: {self.max_diff(x1[i], x2[i])}, rtol: {rtol}, atol: {atol})")

    def compare_backward(self, model1, dx1, model2, dx2):
        assert len(dx1) == len(dx2), "dx1 and dx2 should have the same length"
        if verbose_test():
            print()
            print(f"Comparing dx of both models...")
        for i, layer in reversed(list(enumerate(model2.layers[2:], 2))):
            # Skip test on layers that behave randomly
            if layer.canonical_name != "Dropout":
                rtol, atol = self.get_tolerance(layer)
                if dx1[i].shape == dx2[i].shape:
                    allclose = np.allclose(dx1[i], dx2[i], rtol=rtol, atol=atol)
                else:
                    warnings.warn(f"dx shape on both models for {layer.canonical_name_with_id} differ:"
                                  f" [dx1.shape: {dx1[i].shape}, dx2.shape: {dx2[i].shape}]")
                    # Try flattening both
                    allclose = np.allclose(dx1[i].flatten(), dx2[i].flatten(), rtol=rtol, atol=atol)
                self.assertTrue(allclose,
                                f"Backward result from layer {layer.canonical_name_with_id} differ"
                                f" (max diff: {self.max_diff(dx1[i], dx2[i])}, rtol: {rtol}, atol: {atol})")

    def do_test_model(self, model_name):
        """
        Compares results between a model that uses I2C and other that uses ConvGemm
        """
        console = Console(force_terminal=not verbose_test())
        with console.status("", spinner="dots"):

            # Model 1 forward
            model1, loss_func1 = self.get_model1_and_loss_func(model_name)
            model1.mode = TRAIN_MODE
            if verbose_test():
                print()
                print_with_header(f"Model {model1.model_name} 1 forward pass")
            x1 = self.do_model1_forward_pass(model1)

            # Model 2 forward
            model2 = self.get_model2(model_name)
            model2.mode = TRAIN_MODE
            self.copy_weights_and_biases(model1, model2)
            if verbose_test():
                print_with_header(f"Model {model2.model_name} 2 forward pass")
            x2 = self.do_model2_forward_pass(model2, x1)

            # Compare forward results
            self.compare_forward(model1, x1, model2, x2)

            # Model 1 backward
            if verbose_test():
                print_with_header(f"Model {model1.model_name} 1 backward pass")
            first_dx = self.get_first_dx(model1, loss_func1, x1[-1])
            dx1 = self.do_model1_backward_pass(model1, first_dx)

            # Model 2 backward
            if verbose_test():
                print_with_header(f"Model {model2.model_name} 2 backward pass")
            dx2 = self.do_model2_backward_pass(model2, dx1)

            # Compare backward results
            self.compare_backward(model1, dx1, model2, dx2)

    def test_alexnet(self):
        f"""
        Compares results between an Alexnet model {self.model1_desc} and other {self.model1_desc}
        """
        self.do_test_model("alexnet_cifar10")

    def test_vgg11(self):
        f"""
        Compares results between a VGG-11 BN model {self.model1_desc} and other {self.model1_desc}
        """
        self.do_test_model("vgg11_cifar10")

    def test_vgg16bn(self):
        f"""
        Compares results between a VGG-16 BN model {self.model1_desc} and other {self.model1_desc}
        """
        self.do_test_model("vgg16bn_cifar10")

    def test_resnet34(self):
        f"""
        Compares results between a Densenet model {self.model1_desc} and other {self.model1_desc}
        """
        self.do_test_model("resnet34_cifar10")

    def test_densenet(self):
        f"""
        Compares results between a Densenet model {self.model1_desc} and other {self.model1_desc}
        """
        self.do_test_model("densenet_cifar10")


if __name__ == '__main__':
    try:
        Model()
    except NameError:
        sys.exit(-1)
    unittest.main()
