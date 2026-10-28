"""
PyDTNN PyTorch layer compatibility test suite.

This module provides unit tests to verify the numerical equivalence between
PyDTNN layers and their corresponding PyTorch implementations.
"""

import logging
from typing import Any
import unittest

import numpy as np
import torch
from torch.nn import Module

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.arctanh import Arctanh
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.model import Model
from pydtnn.model.base import ModelMode
from pydtnn.tests.abstract.pytorch import ParamsTestTorchAbstract, PytorchModelTestCaseAbstract
from pydtnn.utils.constants import Parameters
from pydtnn.utils.tensor import TensorFormat, format_transpose
from pydtnn.utils.tensor_array import TensorArray

__all__ = (
    "PytorchLayerParams",
    "PytorchLayerTestCase",
    "TorchAdditionBlock",
    "TorchArcTanH",
    "TorchConcatenationBlock",
)

logger = logging.getLogger(__name__)


# from torch.testing._internal.common_utils import _numpy_to_torch_dtype_dict
numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.uint16: torch.uint16,
    np.uint32: torch.uint32,
    np.uint64: torch.uint64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}

# Constant values

GRAD_EQUIVALENCES: dict[str, str] = {
    Parameters.WEIGHTS: "weight",
    Parameters.BIASES: "bias",
}


class PytorchLayerParams(ParamsTestTorchAbstract):
    """Configuration parameters for PyTorch layer tests."""

    def __init__(self) -> None:
        """Initializes test parameters based on provided dimensions."""
        super().__init__()
        self.num_epochs = 30
        self.tensor_format = TensorFormat.NCHW
        self.synthetic_input_shape = (3, 32, 32)
        self.synthetic_output_shape = (10,)
        self.dtype = np.dtype(np.float64)
        self.optimizer_name = "adam"
        self.optimizer_nesterov = True
        self.optimizer_decoupled_decay = False
        self.optimizer_momentum = 1
        self.learning_rate = 1e-8
        self.optimizer_beta1 = 0.99
        self.optimizer_beta2 = 0.999
        self.optimizer_epsilon = 1e-08
        self.optimizer_decay = 0.5
        self.backend = "cpu"
        # torch.set_default_dtype(numpy_to_torch_dtype_dict[self.dtype.type])
        # -----
        self.conv2d_in_c_torch = self.synthetic_input_shape[0]
        self.conv2d_n_filters = 2
        self.conv2d_filter_shape = (4, 4)
        self.conv2d_padding = 0
        self.conv2d_stride = 1
        self.conv2d_dilation = 1
        self.conv2d_depthwise_padding = 1

    def asdict(self) -> dict[str, Any]:
        """Returns the parameters as a dictionary."""
        return self.__dict__
# PyTorch models


PARAMS = PytorchLayerParams()


class TorchArcTanH(torch.nn.Module):
    """PyTorch implementation of the Arctanh activation function."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the Arctanh module."""
        super().__init__(*args, **kwargs)
        self.arc_tanh = torch.atanh
        self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Arctanh activation."""
        x = self.flatten(x)
        x = self.arc_tanh(x)
        return x


class TorchAdditionBlock(torch.nn.Module):
    """PyTorch model representing an addition block for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the addition block with two parallel convolutional paths."""
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=PARAMS.conv2d_in_c_torch,
                out_channels=PARAMS.conv2d_n_filters,
                kernel_size=PARAMS.conv2d_filter_shape,
                padding=PARAMS.conv2d_padding,
                stride=PARAMS.conv2d_stride,
                dilation=PARAMS.conv2d_dilation,
            ),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=PARAMS.conv2d_in_c_torch,
                out_channels=PARAMS.conv2d_n_filters,
                kernel_size=PARAMS.conv2d_filter_shape,
                padding=PARAMS.conv2d_padding,
                stride=PARAMS.conv2d_stride,
                dilation=PARAMS.conv2d_dilation,
            )
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the sum of the two block outputs."""
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = x1 + x2
        x = self.flatten(x)
        return x


class TorchConcatenationBlock(torch.nn.Module):
    """PyTorch model representing a concatenation block for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the concatenation block with two parallel convolutional paths."""
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=PARAMS.conv2d_in_c_torch,
                out_channels=PARAMS.conv2d_n_filters,
                kernel_size=PARAMS.conv2d_filter_shape,
                padding=PARAMS.conv2d_padding,
                stride=PARAMS.conv2d_stride,
                dilation=PARAMS.conv2d_dilation,
            ),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=PARAMS.conv2d_in_c_torch,
                out_channels=PARAMS.conv2d_n_filters,
                kernel_size=PARAMS.conv2d_filter_shape,
                padding=PARAMS.conv2d_padding,
                stride=PARAMS.conv2d_stride,
                dilation=PARAMS.conv2d_dilation,
            )
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the concatenation of the two block outputs along the channel dimension."""
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.flatten(x)
        return x


class TorchDepthPointConv(torch.nn.Module):
    """PyTorch model representing a depthwise-pointwise convolution sequence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the depthwise and pointwise convolution layers."""
        super().__init__(*args, **kwargs)

        input_filt = PARAMS.conv2d_in_c_torch
        output_filt = PARAMS.conv2d_n_filters
        stride = PARAMS.conv2d_stride
        padding = PARAMS.conv2d_depthwise_padding

        self.conv_depth = torch.nn.Conv2d(
            in_channels=input_filt,
            out_channels=input_filt,
            kernel_size=PARAMS.conv2d_filter_shape,
            stride=stride,
            padding=padding,
            groups=input_filt,
        )
        self.conv_point = torch.nn.Conv2d(
            in_channels=input_filt,
            out_channels=output_filt,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the depthwise followed by the pointwise convolution."""
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.flatten(x)
        return x


class PytorchLayerTestCase(PytorchModelTestCaseAbstract):
    """Base test case class for verifying PyDTNN layers against PyTorch."""

    # NOTE: Delete parent test to prevent re-export and re-testing
    global PytorchModelTestCaseAbstract
    del PytorchModelTestCaseAbstract

    rtol_default = 1e-4  # 1e-10  # 1e-4
    atol_default = 1e-4  # 1e-10  # 1e-4
    rtol_dict = {}  # BatchNormalization: 1e-5, Conv2D: 1e-5,
    atol_dict = {}  # BatchNormalization: 1e-4, Conv2D: 1e-5,

    def setUp(self) -> None:
        """Sets up the test environment."""
        super().setUp()
        torch.manual_seed(0)

    # Initialization methods

    params = PytorchLayerParams()

    @staticmethod
    def initialize_pydtnn_model(
        list_layers: list[Layerable],
        torch_model: torch.nn.Module,
        params: PytorchLayerParams = params,
    ) -> Model:
        """Initializes a PyDTNN model with the provided layers."""
        model = Model(**params.asdict())
        model.add(Identity(params.synthetic_input_shape))
        model.add_layers(list_layers)
        model.add(LogSoftmax())
        model.mode = ModelMode.TRAIN
        model._model_init()
        PytorchLayerTestCase.copy_grad_vars(model, torch_model)
        return model

    @staticmethod
    def _copy_grad_vars(
        grad: np.ndarray, grad_var: str, torch_layer: torch.nn.Module
    ) -> None:
        """Copies gradient variables from PyDTNN to PyTorch layers."""
        if grad is not None:
            torch_grad_var = GRAD_EQUIVALENCES[grad_var]
            torch_grad: torch.Tensor = getattr(torch_layer, torch_grad_var)
            torch_grad.copy_(
                torch.from_numpy(grad.reshape(torch_grad.shape, copy=True))
                .to(torch.device("cpu"))
                .float()
            )

    @staticmethod
    def copy_grad_vars(model_pydtnn: Model, model_torch: torch.nn.Module) -> None:
        """Synchronizes model parameters between PyDTNN and PyTorch."""
        layers = [
            layer
            for layer in model_pydtnn.get_all_layers()
            if not isinstance(layer, AbstractBlockLayer)
        ]

        if isinstance(layers[0], Identity):
            layers.pop(0)

        torch_layers = list()
        list_children = list(model_torch.children())
        if len(list_children) == 0:
            torch_layers.append(model_torch)
        else:
            for module in list_children:
                if isinstance(module, torch.nn.Sequential):
                    torch_layers.extend(module.children())
                else:
                    torch_layers.append(module)

        with torch.no_grad():
            for i in range(len(torch_layers)):
                layer = layers[i]
                torch_layer = torch_layers[i]
                match layer:
                    case Conv2D() | Conv2DDepthwise():
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            if grad_var == Parameters.WEIGHTS and grad is not None:
                                grad = format_transpose(
                                    grad,
                                    {TensorFormat.NHWC: "ihwo", TensorFormat.NCHW: "oihw"}[
                                        model_pydtnn.tensor_format
                                    ],
                                    "oihw")
                            PytorchLayerTestCase._copy_grad_vars(grad, grad_var, torch_layer)
                    case _:
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            PytorchLayerTestCase._copy_grad_vars(grad, grad_var, torch_layer)

    def compare_forward_mod_values(
        self,
        values_torch: list[tuple[Module, torch.Tensor]],
        values_pydtnn: list[tuple[Layerable, np.ndarray | TensorArray]],
    ) -> tuple[list[tuple[Module, torch.Tensor]], list[tuple[Layerable, np.ndarray | TensorArray]]]:
        """Method to pick the right forward's values to test."""
        return (values_torch, values_pydtnn)

    def compare_backward_mod_values(
        self,
        values_torch: list[tuple[Module, tuple[torch.Tensor, ...] | torch.Tensor]],
        values_pydtnn: list[tuple[Layerable, np.ndarray | TensorArray]],
    ) -> tuple[list[tuple[Module, tuple[torch.Tensor, ...] | torch.Tensor]],
               list[tuple[Layerable, np.ndarray | TensorArray]]]:
        """Method to pick the right backward's values to test."""
        return (values_torch, values_pydtnn)

    def compare_grad_vars_mod_values(
        self,
        values_torch: dict[int, dict[str, tuple[torch.nn.Module, np.ndarray | None]]],
        values_pydtnn: dict[int, tuple[Layerable, dict[str, np.ndarray | None]]],
    ) -> tuple[dict[int, dict[str, tuple[torch.nn.Module, np.ndarray | None]]],
               dict[int, tuple[Layerable, dict[str, np.ndarray | None]]]]:
        """Method to pick the right grad_vars's values to test."""
        return (values_torch, values_pydtnn)

    def compare_optimizer_mod_values(
        self,
        torch_params: dict[int, dict[str, np.ndarray | None]],
        pydtnn_params: dict[int, dict[str, np.ndarray | None]]
    ) -> tuple[dict[int, dict[str, np.ndarray | None]], dict[int, dict[str, np.ndarray | None]]]:
        """Method to pick the optimizer's values to test."""
        return (torch_params, pydtnn_params)

    def compare_forward(
        self,
        values_torch: list[tuple[Module, torch.Tensor]],
        values_pydtnn: list[tuple[Layerable, np.ndarray | TensorArray]],
        round: int
    ) -> None:
        """Method to adapt the forward outputs to the test."""
        values_torch, values_pydtnn = self.compare_forward_mod_values(values_torch, values_pydtnn)
        return super().compare_forward(values_torch, values_pydtnn, round)

    def compare_backward(
        self,
        values_torch: list[tuple[Module, tuple[torch.Tensor, ...] | torch.Tensor]],
        values_pydtnn: list[tuple[Layerable, np.ndarray | TensorArray]],
        round: int,
        off_set_pydtnn: int = 0
    ) -> None:
        """Method to adapt the backward outputs to the test."""
        values_torch, values_pydtnn = self.compare_backward_mod_values(values_torch, values_pydtnn)
        return super().compare_backward(values_torch, values_pydtnn, round, off_set_pydtnn=off_set_pydtnn)

    def compare_grad_vars(
        self,
        torch_model: Module,
        pydtnn_model: Model,
        round: int
    ) -> None:
        """Method to adapt the grad_vars to the test."""
        torch_gradients = self.get_torch_grad_parameters_values(torch_model)
        pydtnn_gradients = self.get_pydtnn_grad_vars_values(pydtnn_model)
        torch_gradients, pydtnn_gradients = self.compare_grad_vars_mod_values(torch_gradients, pydtnn_gradients)
        return super()._compare_grad_vars(torch_gradients, pydtnn_gradients, round)

    def compare_optimizer_parameters(
        self, torch_model: Module, pydtnn_model: Model, round: int
    ) -> None:
        """Method to adapt the optimizer parameters's to the test."""
        torch_params = self.get_torch_parameters_values(torch_model)
        pydtnn_params = self.get_pydtnn_vars_values(pydtnn_model)
        torch_params, pydtnn_params = self.compare_optimizer_mod_values(torch_params, pydtnn_params)
        return super().compare_optimizer_parameters(torch_model, pydtnn_model, round)

    ########################
    # Unitary Test methods #
    ########################
    @unittest.skip("Work in progress.")
    def test_addition_block(self) -> None:
        """Tests AdditionBlock layer."""
        pydtnn_layers: list[Layerable] = [
            AdditionBlock(
                [
                    Conv2D(
                        nfilters=PARAMS.conv2d_n_filters,
                        filter_shape=PARAMS.conv2d_filter_shape,
                        padding=PARAMS.conv2d_padding,
                        stride=PARAMS.conv2d_stride,
                        dilation=PARAMS.conv2d_dilation,
                    ),
                ],
                [
                    Conv2D(
                        nfilters=PARAMS.conv2d_n_filters,
                        filter_shape=PARAMS.conv2d_filter_shape,
                        padding=PARAMS.conv2d_padding,
                        stride=PARAMS.conv2d_stride,
                        dilation=PARAMS.conv2d_dilation,
                    )
                ],
            ),
            Flatten()
        ]

        model_torch = TorchAdditionBlock()
        model_pydtnn = PytorchLayerTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params, torch_model=model_torch
        )
        self.do_test_model(
            model_pydtnn=model_pydtnn,
            model_torch=model_torch,
            model_name="AdditionBlock",
        )

    @unittest.skip("Work in progress.")
    def test_concatenation_block(self) -> None:
        """Tests ConcatenationBlock layer."""
        pydtnn_layers: list[Layerable] = [
            ConcatenationBlock(
                [
                    Conv2D(
                        nfilters=PARAMS.conv2d_n_filters,
                        filter_shape=PARAMS.conv2d_filter_shape,
                        padding=PARAMS.conv2d_padding,
                        stride=PARAMS.conv2d_stride,
                        dilation=PARAMS.conv2d_dilation,
                    ),
                ],
                [
                    Conv2D(
                        nfilters=PARAMS.conv2d_n_filters,
                        filter_shape=PARAMS.conv2d_filter_shape,
                        padding=PARAMS.conv2d_padding,
                        stride=PARAMS.conv2d_stride,
                        dilation=PARAMS.conv2d_dilation,
                    )
                ],
            ),
            Flatten()
        ]
        model_torch = TorchConcatenationBlock()
        model_pydtnn = PytorchLayerTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params, torch_model=model_torch
        )
        self.do_test_model(
            model_pydtnn=model_pydtnn,
            model_torch=model_torch,
            model_name="ConcatenationBlock"
        )

    # @unittest.skip("Work in progress.")
    def test_arctanh(self) -> None:
        """Tests Arctanh activation."""
        # NOTE: Domain ArcTanH: Real numbers between "]-1, 1["
        pydtnn_layers: list[Layerable] = [Arctanh(), Flatten()]
        model_torch = TorchArcTanH()
        model_pydtnn = PytorchLayerTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params, torch_model=model_torch
        )
        self.do_test_model(model_pydtnn=model_pydtnn, model_torch=model_torch,
                           model_name="Arctanh", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_depthwise_pointwise(self) -> None:
        """Tests Depthwise and Pointwise convolution sequence."""
        input_filt = PARAMS.conv2d_in_c_torch
        output_filt = PARAMS.conv2d_n_filters

        pydtnn_layers: list[Layerable] = [
            Conv2DDepthwise(
                nfilters=input_filt,
                stride=PARAMS.conv2d_stride,
                filter_shape=PARAMS.conv2d_filter_shape,
                padding=PARAMS.conv2d_depthwise_padding,
            ),
            Conv2DPointwise(nfilters=output_filt),
            Flatten()
        ]
        model_torch = TorchDepthPointConv()
        model_pydtnn = PytorchLayerTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params, torch_model=model_torch
        )

        def compare_forward_mod_values(
            values_torch: list[tuple[Module, torch.Tensor]],
            values_pydtnn: list[tuple[Layerable, np.ndarray | TensorArray]],
        ) -> tuple[list[tuple[Module, torch.Tensor]],
                   list[tuple[Layerable, np.ndarray | TensorArray]]]:
            return ([values_torch[1]], [values_pydtnn[1]])

        def compare_backward_mod_values(
            values_torch: list[tuple[Module, tuple[torch.Tensor, ...] | torch.Tensor]],
            values_pydtnn: list[tuple[Layerable, np.ndarray | TensorArray]],
        ) -> tuple[list[tuple[Module, tuple[torch.Tensor, ...] | torch.Tensor]],
                   list[tuple[Layerable, np.ndarray | TensorArray]]]:
            return ([values_torch[-1]], [values_pydtnn[-1]])

        self.compare_forward_mod_values = compare_forward_mod_values
        self.compare_backward_mod_values = compare_backward_mod_values

        self.do_test_model(model_pydtnn=model_pydtnn, model_torch=model_torch, model_name="Depthwise-Pointwise")
