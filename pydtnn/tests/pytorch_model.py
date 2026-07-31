"""Tests for verifying model behavior and consistency across different data types."""

import logging
import unittest
import warnings

import numpy as np
import torch
import torchvision.models as torch_models

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.abstract.layer import LayerError
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model as PyDTNN_Model
from pydtnn.tests.abstract.base import Params, verbose_test, TestCase
from pydtnn.utils import print_with_header, rand
from pydtnn.utils.pytorch import from_pytorch
from pydtnn.utils.tensor import TensorFormat

type PyTorch_Model = torch.nn.Module

__all__ = ("PytorchModelTestCase",)

logger = logging.getLogger(__name__)


class ResNet14Like(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1_0 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
        )
        self.layer1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
        )
        self.layer2_0 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
        )
        self.layer3_0 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
                torch.nn.BatchNorm2d(
                    512, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
                ),
            ),
        )
        self.layer3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
        )
        self.layer4_0 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
        )
        self.layer4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
        )

        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.layer1 = torch.nn.Sequential(self.layer1_0, self.layer1_1)
        self.layer2 = torch.nn.Sequential(self.layer2_0)
        self.layer3 = torch.nn.Sequential(self.layer3_0, self.layer3_1)
        self.layer4 = torch.nn.Sequential(self.layer4_0, self.layer4_1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=10, bias=True),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Feature layers
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Classifier layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 10),
            torch.nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def replace_layer(
    module: torch.nn.Module, layer_to_replace: type[torch.nn.Module]
) -> None:
    """
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/7
    """

    # iterate through immediate child modules
    list_children = list(module.named_children())
    for name, immediate_child_module in list_children:
        setattr(module, name, replace_layer(immediate_child_module, layer_to_replace))

    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    if isinstance(module, layer_to_replace):
        if verbose_test():
            print(f"Removed: {module}")
        return torch.nn.Identity()
    else:
        return module


class PytorchModelTestCase(TestCase):
    """Tests that two models with different parameters lead to the same results"""

    # NOTE: Delete parent test to prevent re-export and re-testing
    global TestCase
    del TestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model2_desc}
    model1_desc = "using PyTorch"
    model2_desc = "using PyDTNN"

    rtol_default = 1e-4
    atol_default = 1e-5
    rtol_dict = {
        AdditionBlock: 5e-3,
        ConcatenationBlock: 1e-1,
        BatchNormalization: 1e-5,
        Conv2D: 1e-4,
    }
    atol_dict = {
        AdditionBlock: 5e-3,
        ConcatenationBlock: 1e-1,
        Conv2D: 1e-5,
        BatchNormalization: 1e-4,
    }

    def setUp(self) -> None:
        """Sets up the test environment."""
        super().setUp()
        torch.manual_seed(0)
    # Initialization methods

    params = Params()
    params.num_epochs = 3
    params.tensor_format = TensorFormat.NCHW
    params.synthetic_input_shape = (3, 32, 32)
    params.synthetic_output_shape = (10,)

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

    @staticmethod
    def get_model_torch_and_loss_func(
        model_name: str,
    ) -> tuple[PyTorch_Model, torch.nn.modules.loss._Loss]:
        """
        Initializes a model and its corresponding loss function.

        Args:
            model_name: Name of the model to initialize.
            overwrite_params: Optional dictionary to override default parameters.

        Returns:
            A tuple containing the initialized PyTorch_Model and Loss objects.
        """
        # PyTorch Model.
        params = PytorchModelTestCase.params
        params.model_name = model_name

        match model_name:
            case "simplecnn":
                torch_model = SimpleCNN()
            case "resnet14like":
                torch_model = ResNet14Like()
            case "resnet50":
                torch_model = torch_models.resnet50(weights=torch_models.ResNet50_Weights.IMAGENET1K_V1)
                torch_model.fc = torch.nn.Sequential(  # pyright: ignore[reportAttributeAccessIssue]
                    # torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(
                        in_features=torch_model.fc.in_features,
                        out_features=params.synthetic_output_shape[0],
                    ),
                    torch.nn.LogSoftmax(dim=1),
                )
            case _:
                raise ValueError(f"Unknown model {model_name!r}!")

        replace_layer(torch_model, layer_to_replace=torch.nn.Dropout)

        return torch_model, torch.nn.NLLLoss()

    def get_model_pydtnn(self, model_pytorch: PyTorch_Model) -> PyDTNN_Model:
        """
        Constructs and returns a model instance configured with float64 precision.

        Args:
            model_name: The name of the model to instantiate.
            overwrite_params: Optional dictionary of parameters to override defaults.

        Returns:
            A PyDTNN_Model instance configured for float64.

        Raises:
            unittest.SkipTest: If the model is incompatible with the dataset.
        """
        # PyDTNN Model
        params = PytorchModelTestCase.params
        params.loss_func_name = "negative_log_likelihood"
        params.model_name = ""
        # Begin of params configuration
        params_dict = vars(params)
        try:
            model_pydtnn = PyDTNN_Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(
                f"PyDTNN_Model incompatible with {params_dict['dataset_name']}"
            ) from exc

        if verbose_test():
            print(model_pytorch)

        layers = from_pytorch(params.synthetic_input_shape, model_pytorch)
        model_pydtnn.add_layers(layers)
        model_pydtnn._model_init()
        model_pydtnn.mode = model_pydtnn.Mode.TRAIN

        if verbose_test():
            for layer in model_pydtnn.layers:
                print(layer)

        # capa = model_pydtnn.layers[-1]

        return model_pydtnn

    def do_model1_forward_pass(self, model1: PyTorch_Model, x0: torch.Tensor) -> list[torch.Tensor]:
        """
        Performs a forward pass for Model 1.

        Args:
            model1: The model instance.
            x0: Initial input list.

        Returns:
            List of outputs after each layer.
        """
        x1: list[torch.Tensor] = [x0]
        # TODO: Get all layers in torch and iterate over those layers.
        x1.append(model1(x1[0].clone()))
        return x1

    def do_model2_forward_pass(self, model2: PyDTNN_Model, x0: np.ndarray) -> list[np.ndarray]:
        """
        Performs a forward pass for Model 1.

        Args:
            model1: The model instance.
            x0: Initial input list.

        Returns:
            List of outputs after each layer.
        """
        x1: list[np.ndarray] = [x0]
        # TODO: Get all layers in torch and iterate over those layers.
        for layer in model2.layers:
            x1.append(layer.forward(x1[-1].copy()))
        return x1

    def do_pytorch_model_loss(
        self, loss_func: torch.nn.modules.loss._Loss, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Method execute the pytorch's loss"""
        loss: torch.Tensor = loss_func(x, y)
        return loss

    def do_pydtnn_model_loss(
        self, model: PyDTNN_Model, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Method execute the pydtnn's loss"""
        loss, dx = model.loss_func.compute(x.copy(), y, model.batch_size)
        return loss, dx

    def do_pytorch_model_backward_pass(
        self, _model1: PyTorch_Model, loss: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Performs a forward pass for Model 1.

        Args:
            _model1: The PyTorch model
            loss: the loss' function output

        Returns:
            Nothing special yet.
        """
        # TODO: mover el loss a una función a parte (para compararlas)
        # dx: list[torch.Tensor] = []
        with warnings.catch_warnings(action="ignore"):
            loss.backward()

        dx = loss
        # TODO: Get all layers in torch and iterate over those layers.
        return [dx]

    def do_pydtnn_model_backward_pass(
        self, model2: PyDTNN_Model, dx: np.ndarray
    ) -> list[np.ndarray]:
        """
        Performs a forward pass for PyDTNN's Model.

        Args:
            model: The model instance.
            dx: Initial gradient.

        Returns:
            List of gradients after each layer.
        """
        # TODO: Get all layers in torch and iterate over those layers.
        dx1 = list[np.ndarray]()
        # NOTE: Since we're iterating the layers from the end do the start,
        #   we're inserting the data in the first element so at the end the gradients
        #   will be at the same index as it's layer.
        dx1.insert(0, dx)
        for layer in reversed(model2.layers):
            dx1.insert(0, layer.backward(dx1[0].copy()))
        return dx1

    def do_pytorch_model_optimizer_pass(
        self, _model1: PyTorch_Model, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Performs a forward pass for Model 1.

        Args:
            _model1: The PyTorch model
            loss: the loss' function output

        Returns:
            Nothing special yet.
        """
        # TODO: mover el loss a una función a parte (para compararlas)
        # dx: list[torch.Tensor] = []
        optimizer.step()

    def do_pydtnn_model_optimizer_pass(
        self, model2: PyDTNN_Model
    ) -> None:
        """
        Performs a forward pass for PyDTNN's Model.

        Args:
            model: The model instance.
            dx: Initial gradient.

        Returns:
            List of gradients after each layer.
        """
        for layer in model2.layers:
            layer.update_weights(model2.optimizer, update=True, sync=False)

    def compare_forward(
        self, model_pydtnn: PyDTNN_Model, x_torch: list[torch.Tensor], x_pydtnn: list[np.ndarray]
    ) -> None:
        """
        Compares the forward pass outputs of two models.

        Args:
            model1 (PyTorch_Model): PyTorch model.
            x1 (list[torch.Tensor]): Forward pass outputs of model 1.
            model2 (PyDTNN_Model): PyDTNN model.
            x2 (list[np.ndarray]): Forward pass outputs of model 2.
        """
        # assert len(x1) == len(x2), "x1 and x2 should have the same length"
        if verbose_test():
            print("Comparing outputs of both models...")

        pytorch_last_layer = x_torch[-1]
        pydtnn_last_layer = x_pydtnn[-1]
        layer = model_pydtnn.layers[-1]

        pytorch_as_numpy = pytorch_last_layer.numpy(force=True)

        rtol, atol = self.get_tolerance(layer)
        self.assertTrue(
            pytorch_as_numpy.size == pydtnn_last_layer.size,
            f"Both tensors doesn't have the same number of elements "
            f"({pytorch_as_numpy.size=} != {pydtnn_last_layer.size=})",
        )
        self.assertTrue(
            np.allclose(pytorch_as_numpy, pydtnn_last_layer, rtol=rtol, atol=atol),
            f"Forward result from layers {layer.name_with_id} differ "
            f"({self.print_stats(pytorch_as_numpy, pydtnn_last_layer, rtol, atol)})",
        )

    def compare_backward(
        self,
        _model1: PyTorch_Model,
        _dx1: list[torch.Tensor],
        _model2: PyDTNN_Model,
        _dx2: list[np.ndarray],
    ) -> None:
        """Compares the backward pass gradients of two models.

        Due the differences of implementation, right now it's not possible to compare the both gradients.

        Args:
            model1: First model.
            dx1: Backward pass gradients of model 1.
            model2: Second model.
            dx2: Backward pass gradients of model 2.
        """
        pass

    @staticmethod
    def target_pydtnn2torch_format(y_pydtnn: np.ndarray) -> np.ndarray:
        """Convert PyDTNN output shape to PyTorch's format"""
        return np.argmax(y_pydtnn, axis=1)

    def do_test_model(self, model_name: str) -> None:
        """
        Executes the full comparison test for a given model.

        Args:
            model_name: Name of the model to test.
        """

        model_torch, loss_func_torch = self.get_model_torch_and_loss_func(model_name)
        model_pydtnn = self.get_model_pydtnn(model_torch)
        model_pydtnn.mode = PyDTNN_Model.Mode.TRAIN

        params = PytorchModelTestCase.params
        input_shape = params.synthetic_input_shape
        output_shape = params.synthetic_output_shape[0]

        for i in range(params.num_epochs):
            if verbose_test():
                print(f"Round {i + 1}/{params.num_epochs}")

            x_pydtnn = np.asarray(
                rand.random((params.batch_size, *input_shape)), dtype=params.dtype, order="C"
            )
            y_pydtnn = np.ones((params.batch_size, output_shape), dtype=params.dtype)

            x_torch = torch.from_numpy(x_pydtnn).to(torch.device("cpu")).float()
            y_torch = torch.from_numpy(self.target_pydtnn2torch_format(y_pydtnn)).to(torch.device("cpu")).long()

            # --- FORWARD ---
            if verbose_test():
                print_with_header(f"Model {model_name} 1 forward pass")
            x_torch = self.do_model1_forward_pass(model_torch, x_torch)

            if verbose_test():
                print_with_header(f"Model {model_pydtnn.model_name} 2 forward pass")
            x_pydtnn = self.do_model2_forward_pass(model_pydtnn, x_pydtnn)

            # Compare forward results
            self.compare_forward(model_pydtnn, x_torch, x_pydtnn)

            # --- LOSS ---
            loss_torch = self.do_pytorch_model_loss(loss_func_torch, x_torch[-1], y_torch)
            _loss_pydtnn, dx_pydtnn = self.do_pydtnn_model_loss(
                model_pydtnn, x_pydtnn[-1], y_pydtnn
            )
            # TODO: Add loss comparation

            # --- BACKWARD ---
            # Model 1 backward
            if verbose_test():
                print_with_header(f"Model {model_torch} 1 backward pass")

            dx_torch = self.do_pytorch_model_backward_pass(model_torch, loss_torch)

            # Model 2 backward
            if verbose_test():
                print_with_header(f"Model {model_pydtnn.model_name} 2 backward pass")
            dx_pydtnn = self.do_pydtnn_model_backward_pass(model_pydtnn, dx_pydtnn)

            # Compare backward results
            self.compare_backward(model_torch, dx_torch, model_pydtnn, dx_pydtnn)

            # TODO: Add optimizers
            # optimizer.step()  # Torch

    @unittest.skip("Large model")
    def test_renset50(self) -> None:
        """Compares results between an ResNet50 model using a PyTorch model and other a PyDTNN one."""
        self.do_test_model("resnet50")

    def test_resnet14like(self) -> None:
        """Compares results between an ResNet14_like model using a PyTorch model and other a PyDTNN one."""
        self.do_test_model("resnet14like")

    def test_simplecnn(self) -> None:
        """Compares results between an SimpleCNN model using a PyTorch model and other a PyDTNN one."""
        self.do_test_model("simplecnn")
