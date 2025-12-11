import unittest
import warnings

import numpy as np

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.losses.loss import Loss, select as select_loss
from pydtnn.layers.layer import LayerError
from pydtnn.model import Model
from pydtnn.tests.abstract.common import verbose_test

from pydtnn.layers.layer import LayerBase
from pydtnn.tests.abstract.common import Params, TestCase
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils import print_with_header, random


class ModelCommonTestCase(TestCase):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using A"
    model2_desc = "using B"

    rtol_default = 1e-4
    atol_default = 1e-5
    rtol_dict = {
        AdditionBlock: 1e-4,
        ConcatenationBlock: 1e-1,
        BatchNormalization: 1e-5,
        Conv2D: 1e-4,
    }
    atol_dict = {
        AdditionBlock: 5e-4,
        ConcatenationBlock: 1e-1,
        Conv2D: 1e-5,
        BatchNormalization: 1e-4,
    }

    def get_tolerance(self, layer: LayerBase) -> tuple[float, float]:
        rtol = self.rtol_default
        for cls, tol in self.rtol_dict.items():
            if isinstance(layer, cls):
                rtol = tol
                break

        atol = self.atol_default
        for cls, tol in self.atol_dict.items():
            if isinstance(layer, cls):
                atol = tol
                break

        # NOTE: Revise group layer tolerance
        if isinstance(layer, AbstractBlockLayer):
            for child in layer.children:
                crtol, catol = self.get_tolerance(child)
                rtol += crtol
                atol += catol

        return rtol, atol

    @staticmethod
    def get_model1_and_loss_func(model_name: str, overwrite_params: dict | None = None) -> tuple[Model, Loss]:
        # CPU model with no convGemm
        params = Params()
        # Begin of params configuration
        params.model_name = model_name  # type: ignore
        params.tensor_format = TensorFormat.NHWC.upper()
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        try:
            model1 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        # loss function
        loss_func_name = model1.loss_func_name
        local_batch_size = model1.batch_size
        loss_cls = select_loss(loss_func_name)
        loss_func = loss_cls(shape=(local_batch_size, *model1.layers[-1].shape))
        loss_func.init_backend_from_model(model1)
        loss_func.initialize()
        return model1, loss_func

    def get_model2(self, model_name: str, overwrite_params: dict | None = None) -> Model:
        raise NotImplementedError()

    def copy_weights_and_biases(self, model1: Model, model2: Model):
        """
        Copy weights and biases from Model 1 to Model 2
        """
        model2.import_(model1)

    def get_first_dx(self, model: Model, loss_func: Loss, x: np.ndarray) -> np.ndarray:
        # random y target
        y_targ = np.asarray(random.random(x.shape), dtype=model.dtype, order='C', copy=True)
        # obtain first dx1
        global_batch_size = model.batch_size
        loss, dx = loss_func.compute(x, y_targ, global_batch_size)
        return dx

    def print_stats(self, x1: np.ndarray, x2: np.ndarray, rtol: float, atol: float) -> str:
        diff = x1 - x2
        return '\n' \
               f"\t{rtol=}\n"\
               f"\t{atol=}\n"\
               f"\tmax_diff={np.max(np.abs(diff))}\n" \
               f"\t{x1.max()=}\n" \
               f"\t{x2.max()=}\n" \
               f"\t{diff.max()=}\n" \
               f"\t{x1.min()=}\n" \
               f"\t{x2.min()=}\n" \
               f"\t{diff.min()=}\n" \
               f"\t{x1.std()=}\n" \
               f"\t{x2.std()=}\n" \
               f"\t{diff.std()=}\n" \
               f"\t{x1.mean()=}\n" \
               f"\t{x2.mean()=}\n" \
               f"\t{diff.mean()=}\n"
    # ---

    def do_model1_forward_pass(self, model1: Model, x0: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 1 forward pass
        """
        x1 = [x0[0]]
        for i, layer in enumerate(model1.layers):
            if verbose_test():
                print(layer)
            x1.append(layer.forward(np.asarray(x1[i], dtype=model1.dtype, order="C", copy=True)))
        return x1

    def do_model2_forward_pass(self, model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 forward pass
        """
        x2 = [x1[0]]
        for i, layer in enumerate(model2.layers):
            if verbose_test():
                print(layer)
            x2.append(layer.forward(np.asarray(x1[i], dtype=model2.dtype, order="C", copy=True)))
        return x2

    @staticmethod
    def do_model1_backward_pass(model1: Model, dx0: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 1 backward pass
        """
        dx1 = [dx0[0]]
        for _, layer in reversed(list(enumerate(model1.layers))):
            if verbose_test():
                print(layer)
            dx1.insert(0, layer.backward(np.asarray(dx1[0], dtype=model1.dtype, order="C", copy=True)))
        return dx1

    def do_model2_backward_pass(self, model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 backward pass
        """
        dx2 = [dx1[-1]]
        for i, layer in reversed(list(enumerate(model2.layers))):
            if verbose_test():
                print(layer)
            dx2.insert(0, layer.backward(np.asarray(dx1[i + 1], dtype=model2.dtype, order="C", copy=True)))
        return dx2

    def compare_forward(self, model1: Model, x1: list[np.ndarray], model2: Model, x2: list[np.ndarray]):
        assert len(x1) == len(x2), "x1 and x2 should have the same length"
        if verbose_test():
            print()
            print(f"Comparing outputs of both models...")
        for i, layer in enumerate(model1.layers, 1):
            # Skip test on layers that behave randomly
            if not isinstance(layer, Dropout):
                rtol, atol = self.get_tolerance(layer)
                self.assertTrue(np.allclose(x1[i], x2[i], rtol=rtol, atol=atol),
                                f"Forward result from layers {layer.name_with_id} differ"
                                f" ({self.print_stats(x1[i], x2[i], rtol, atol)})")

    def compare_backward(self, model1: Model, dx1: list[np.ndarray], model2: Model, dx2: list[np.ndarray]):
        assert len(dx1) == len(dx2), f"dx1 and dx2 should have the same length {len(dx1)=}, {len(dx2)=}"
        if verbose_test():
            print("\nComparing outputs shapes.")
            min_dx = min(len(dx1), len(dx2))
            for i in range(min_dx):
                print(f"{i} - {dx1[i].shape=} ||\t{dx2[i].shape=}")
            for i in range(len(dx1) - len(dx2)):
                i = i + min_dx
                print(f"{i} - {dx1[i].shape=}")
            for i in range(len(dx2) - len(dx1)):
                i = i + min_dx
                print(f"{i} - {dx2[i].shape=}")

            print(f"\nComparing dx of both models...")
        for i, layer in enumerate(model2.layers, 0):
            # Skip test on layers that behave randomly
            if not isinstance(layer, Dropout):
                rtol, atol = self.get_tolerance(layer)
                if dx1[i].shape == dx2[i].shape:
                    allclose = np.allclose(dx1[i], dx2[i], rtol=rtol, atol=atol)
                else:
                    warnings.warn(f"dx shape on both models for {layer.name_with_id} differ:"
                                  f" [dx1.shape: {dx1[i].shape}, dx2.shape: {dx2[i].shape}]")
                    # Try flattening both
                    allclose = np.allclose(dx1[i].flatten(), dx2[i].flatten(), rtol=rtol, atol=atol)
                self.assertTrue(allclose,
                                f"Backward result from layer {layer.name_with_id} differ"
                                f" ({self.print_stats(dx1[i], dx2[i], rtol, atol)})")

    def do_test_model(self, model_name: str):
        """
        Compares results between a model that uses I2C and other that uses ConvGemm
        """

        # Model 1 forward
        model1, loss_func1 = self.get_model1_and_loss_func(model_name)
        model1.mode = Model.Mode.TRAIN

        model2 = self.get_model2(model_name)
        model2.mode = Model.Mode.TRAIN
        self.copy_weights_and_biases(model1, model2)

        x = [np.asarray(random.random((model1.batch_size, *model1.layers[0].shape)), dtype=model1.dtype, order='C', copy=True)]

        if verbose_test():
            print()
            print_with_header(f"Model {model1.model_name} 1 forward pass")
        x1 = self.do_model1_forward_pass(model1, x)

        x2 = x1.copy()

        # Model 2 forward
        if verbose_test():
            print_with_header(f"Model {model2.model_name} 2 forward pass")
        x2 = self.do_model2_forward_pass(model2, x2)

        # Compare forward results
        self.compare_forward(model1, x1, model2, x2)

        # Model 1 backward
        if verbose_test():
            print_with_header(f"Model {model1.model_name} 1 backward pass")
        dx = [self.get_first_dx(model1, loss_func1, x1[-1])]

        dx1 = self.do_model1_backward_pass(model1, dx)

        dx2 = dx1.copy()

        # Model 2 backward
        if verbose_test():
            print_with_header(f"Model {model2.model_name} 2 backward pass")
        dx2 = self.do_model2_backward_pass(model2, dx2)

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

    def test_simplecnn(self):
        f"""
        Compares results between a SimpleCNN model {self.model1_desc} and other {self.model1_desc}
        """
        self.do_test_model("simplecnn")
