"""
Module for common model testing utilities in PyDTNN.

Provides a base test class to compare model outputs and gradients across different implementations.
"""

import logging
import unittest

import numpy as np

from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.abstract.layer import Layerable, LayerError
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.losses.abstract.loss import Loss
from pydtnn.model import Model
from pydtnn.tests.abstract.base import Params, TestCase, verbose_test
from pydtnn.utils import print_with_header, rand
from pydtnn.utils.tensor import TensorFormat

__all__ = ("ModelTestCase",)

logger = logging.getLogger(__name__)


class ModelTestCase(TestCase):
    """Tests that two models with different parameters lead to the same results"""

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

    def get_tolerance(self, layer: Layerable) -> tuple[float, float]:
        """
        Calculates the relative and absolute tolerance for a given layer.

        Args:
            layer: The layer instance to check.

        Returns:
            A tuple containing (rtol, atol).
        """
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
    def get_model1_and_loss_func(
        model_name: str, overwrite_params: dict | None = None
    ) -> tuple[Model, Loss]:
        """
        Initializes a model and its corresponding loss function.

        Args:
            model_name: Name of the model to initialize.
            overwrite_params: Optional dictionary to override default parameters.

        Returns:
            A tuple containing the initialized Model and Loss objects.
        """
        # CPU model with no convGemm
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.tensor_format = TensorFormat.NHWC
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        try:
            model1 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(
                f"Model {model_name} incompatible with {params_dict['dataset_name']}"
            ) from exc
        model1._model_init()
        # loss function
        loss_func = model1.loss_func
        return model1, loss_func

    def get_model2(self, model_name: str, overwrite_params: dict | None = None) -> Model:
        """
        Abstract method to retrieve the second model for comparison.

        Args:
            model_name: Name of the model.
            overwrite_params: Optional parameters to override.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError()

    def copy_weights_and_biases(self, model1: Model, model2: Model) -> None:
        """
        Copies weights and biases from Model 1 to Model 2.

        Args:
            model1: Source model.
            model2: Destination model.
        """
        model2.import_(model1)

    def get_first_dx(self, model: Model, loss_func: Loss, x: np.ndarray) -> np.ndarray:
        """
        Computes the initial gradient (dx) for a given model and input.

        Args:
            model: The model instance.
            loss_func: The loss function instance.
            x: Input data array.

        Returns:
            The computed gradient array.
        """
        # random y target
        y_targ = np.asarray(rand.random(x.shape), dtype=model.dtype, order="C").copy()
        # obtain first dx1
        loss_func.model.real_batch_size = model.batch_size
        loss, dx = loss_func.compute(x, y_targ)
        return dx

    def print_stats(self, x1: np.ndarray, x2: np.ndarray, rtol: float, atol: float) -> str:
        """
        Generates a string summary of statistical differences between two arrays.

        Args:
            x1: First array.
            x2: Second array.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.

        Returns:
            A formatted string containing statistics.
        """
        diff = x1 - x2
        return (
            "\n"
            f"\t{rtol=}\n"
            f"\t{atol=}\n"
            f"\tmax_diff={np.max(np.abs(diff))}\n"
            f"\t{x1.max()=}\n"
            f"\t{x2.max()=}\n"
            f"\t{diff.max()=}\n"
            f"\t{x1.min()=}\n"
            f"\t{x2.min()=}\n"
            f"\t{diff.min()=}\n"
            f"\t{x1.std()=}\n"
            f"\t{x2.std()=}\n"
            f"\t{diff.std()=}\n"
            f"\t{x1.mean()=}\n"
            f"\t{x2.mean()=}\n"
            f"\t{diff.mean()=}\n"
            f"\t{np.sum(x1)=}\n"
            f"\t{np.sum(x2)=}\n"
            f"\t{np.sum(diff)=}\n"
        )

    def do_model1_forward_pass(self, model1: Model, x0: list[np.ndarray]) -> list[np.ndarray]:
        """
        Performs a forward pass for Model 1.

        Args:
            model1: The model instance.
            x0: Initial input list.

        Returns:
            List of outputs after each layer.
        """
        x1 = [x0[0]]
        for i, layer in enumerate(model1.layers):
            if verbose_test():
                print(layer)
                print(f"\n{layer.name} - {layer.id} - input {model1.tensor_format=}", end=" - ")
                print(f"{x1[i].max()=}", end=" - ")
                print(f"{x1[i].min()=}", end=" - ")
                print(f"{x1[i].mean()=}", end=" - ")
                print(f"{x1[i].sum()=}", end=" - ")
                print(f"{x1[i].std()=}")
            x1.append(layer.forward(np.asarray(x1[i], dtype=model1.dtype, order="C").copy()).copy())
            if verbose_test():
                print("output", end=" - ")
                print(f"{x1[-1].max()=}", end=" - ")
                print(f"{x1[-1].min()=}", end=" - ")
                print(f"{x1[-1].mean()=}", end=" - ")
                print(f"{x1[-1].sum()=}", end=" - ")
                print(f"{x1[-1].std()=}")
        return x1

    def do_model2_forward_pass(self, model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Performs a forward pass for Model 2.

        Args:
            model2: The model instance.
            x1: Initial input list.

        Returns:
            List of outputs after each layer.
        """
        x2 = [x1[0]]
        for i, layer in enumerate(model2.layers):
            if verbose_test():
                print(layer)
                print(f"\n{layer.name} - {layer.id} - input {model2.tensor_format=}", end=" - ")
                print(f"{x1[i].shape=}", end=" - ")
                print(f"{x1[i].max()=}", end=" - ")
                print(f"{x1[i].min()=}", end=" - ")
                print(f"{x1[i].mean()=}", end=" - ")
                print(f"{x1[i].sum()=}", end=" - ")
                print(f"{x1[i].std()=}")
            x2.append(
                layer.forward(np.asarray(x1[i].copy(), dtype=model2.dtype, order="C").copy()).copy()
            )
            if verbose_test():
                print("output", end=" - ")
                print(f"{x2[-1].max()=}", end=" - ")
                print(f"{x2[-1].min()=}", end=" - ")
                print(f"{x2[-1].mean()=}", end=" - ")
                print(f"{x2[-1].sum()=}", end=" - ")
                print(f"{x2[-1].std()=}")
        return x2

    @staticmethod
    def do_model1_backward_pass(model1: Model, dx0: list[np.ndarray]) -> list[np.ndarray]:
        """
        Performs a backward pass for Model 1.

        Args:
            model1: The model instance.
            dx0: Initial gradient list.

        Returns:
            List of gradients after each layer.
        """
        dx1 = [dx0[0]]
        for i, layer in reversed(list(enumerate(model1.layers))):
            if verbose_test():
                print(layer)
                print(f"\n{layer.name} - {layer.id} - input {model1.tensor_format=}", end=" - ")
                print(f"{dx1[0].max()=}", end=" - ")
                print(f"{dx1[0].min()=}", end=" - ")
                print(f"{dx1[0].mean()=}", end=" - ")
                print(f"{dx1[0].sum()=}", end=" - ")
                print(f"{dx1[0].std()=}")
            dx1.insert(
                0,
                layer.backward(
                    np.asarray(dx1[0].copy(), dtype=model1.dtype, order="C").copy()
                ).copy(),
            )
            if verbose_test():
                print("output", end=" - ")
                print(f"{dx1[0].max()=}", end=" - ")
                print(f"{dx1[0].min()=}", end=" - ")
                print(f"{dx1[0].mean()=}", end=" - ")
                print(f"{dx1[0].sum()=}", end=" - ")
                print(f"{dx1[0].std()=}")
        return dx1

    def do_model2_backward_pass(self, model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Performs a backward pass for Model 2.

        Args:
            model2: The model instance.
            dx1: Initial gradient list.

        Returns:
            List of gradients after each layer.
        """
        dx2 = [dx1[-1]]
        for i, layer in reversed(list(enumerate(model2.layers))):
            if verbose_test():
                print(f"\n{layer}")
                print(f"\n{layer.name} - {layer.id} - input {model2.tensor_format=}", end=" - ")
                print(f"{dx1[i + 1].shape=}", end=" - ")
                print(f"{dx1[i + 1].max()=}", end=" - ")
                print(f"{dx1[i + 1].min()=}", end=" - ")
                print(f"{dx1[i + 1].mean()=}", end=" - ")
                print(f"{dx1[i + 1].sum()=}", end=" - ")
                print(f"{dx1[i + 1].std()=}")
            dx2.insert(
                0,
                layer.backward(
                    np.asarray(dx1[i + 1].copy(), dtype=model2.dtype, order="C").copy()
                ).copy(),
            )
            if verbose_test():
                print("output", end=" - ")
                print(f"{dx2[0].max()=}", end=" - ")
                print(f"{dx2[0].min()=}", end=" - ")
                print(f"{dx2[0].mean()=}", end=" - ")
                print(f"{dx2[0].sum()=}", end=" - ")
                print(f"{dx2[0].std()=}")
        return dx2

    def compare_forward(
        self, model1: Model, x1: list[np.ndarray], model2: Model, x2: list[np.ndarray]
    ) -> None:
        """
        Compares the forward pass outputs of two models.

        Args:
            model1: First model.
            x1: Forward pass outputs of model 1.
            model2: Second model.
            x2: Forward pass outputs of model 2.
        """
        assert len(x1) == len(x2), "x1 and x2 should have the same length"
        if verbose_test():
            print()
            print("Comparing outputs of both models...")
        for i, layer in enumerate(model1.layers, 1):
            # Skip test on layers that behave randomly
            if not isinstance(layer, Dropout):
                rtol, atol = self.get_tolerance(layer)
                self.assertTrue(
                    x1[i].size == x2[i].size,
                    f"Both tensors doesn't have the same number of elements "
                    f"(x1[{i}].size = {x1[i].size} != {x2[i].size} = x2[{i}].size)",
                )
                self.assertTrue(
                    np.allclose(x1[i], x2[i].reshape(x1[i].shape), rtol=rtol, atol=atol),
                    f"Forward result from layers {layer.name_with_id} differ "
                    f"({self.print_stats(x1[i], x2[i], rtol, atol)})",
                )

    def compare_backward(
        self, model1: Model, dx1: list[np.ndarray], model2: Model, dx2: list[np.ndarray]
    ) -> None:
        """
        Compares the backward pass gradients of two models.

        Args:
            model1: First model.
            dx1: Backward pass gradients of model 1.
            model2: Second model.
            dx2: Backward pass gradients of model 2.
        """
        assert len(dx1) == len(dx2), (
            f"dx1 and dx2 should have the same length {len(dx1)=}, {len(dx2)=}"
        )
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

            print("\nComparing dx of both models...")
        for i, layer in enumerate(model2.layers, 0):
            # Skip test on layers that behave randomly
            if not isinstance(layer, Dropout):
                rtol, atol = self.get_tolerance(layer)
                if dx1[i].shape == dx2[i].shape:
                    allclose = np.allclose(dx1[i], dx2[i], rtol=rtol, atol=atol)
                else:
                    logger.warning(
                        f"dx shape on both models for {layer.name_with_id} differ: "
                        f"[dx1.shape: {dx1[i].shape}, dx2.shape: {dx2[i].shape}]"
                    )
                    # Try flattening both
                    self.assertTrue(
                        dx1[i].size == dx2[i].size,
                        f"Both tensors doesn't have the same number of elements "
                        f"(dx1[{i}].size = {dx1[i].size} != {dx2[i].size} = dx2[{i}].size)",
                    )
                    allclose = np.allclose(
                        dx1[i], dx2[i].reshape(dx1[i].shape), rtol=rtol, atol=atol
                    )
                self.assertTrue(
                    allclose,
                    f"Backward result from layer {layer.name_with_id} differ "
                    f"({self.print_stats(dx1[i], dx2[i], rtol, atol)})",
                )

    def do_test_model(self, model_name: str) -> None:
        """
        Executes the full comparison test for a given model.

        Args:
            model_name: Name of the model to test.
        """

        # Model 1 forward
        model1, loss_func1 = self.get_model1_and_loss_func(model_name)
        model1.mode = Model.Mode.TRAIN

        model2 = self.get_model2(model_name)
        model2.mode = Model.Mode.TRAIN
        self.copy_weights_and_biases(model1, model2)

        x = [
            np.asarray(
                rand.random((model1.batch_size, *model1.layers[0].shape)),
                dtype=model1.dtype,
                order="C",
            ).copy()
        ]

        if verbose_test():
            print()
            print_with_header(f"Model {model1.model_name} 1 forward pass")

        model1.real_batch_size = x[0].shape[0]
        model2.real_batch_size = x[0].shape[0]

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

    def test_alexnet(self) -> None:
        """Compares results between an Alexnet model using A and other using B."""
        self.do_test_model("alexnet")

    def test_resnet10(self) -> None:
        """Compares results between a Resnet10 model using A and other using B."""
        self.do_test_model("resnet10")

    def test_densenet21k8(self) -> None:
        """Compares results between a Densenet21 (k=8) model using A and other using B."""
        self.do_test_model("densenet21k8")

    def test_mobilenetv1_tiny(self) -> None:
        """Compares results between a MobileNetV1 (Tiny) model using A and other using B."""
        self.do_test_model("mobilenetv1_tiny")

    def test_mobilenetv2_tiny(self) -> None:
        """Compares results between a MobileNetV2 (Tiny) model using A and other using B."""
        self.do_test_model("mobilenetv2_tiny")
