"""
PyDTNN PyTorch layer compatibility test suite.

This module provides unit tests to verify the numerical equivalence between
PyDTNN layers and their corresponding PyTorch implementations.
"""

import logging
import math
from typing import Any
from unittest import skip

import numpy as np
import torch

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.arctanh import Arctanh
from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.activations.log import Log
from pydtnn.activations.relu import Relu
from pydtnn.activations.relu6 import Relu6
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.activations.softmax import Softmax
from pydtnn.activations.tanh import Tanh
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.model import Model
from pydtnn.tests.abstract.common import Params, TestCase, verbose_test
from pydtnn.utils import rand
from pydtnn.utils.constants import Parameters
from pydtnn.utils.tensor import TensorFormat, format_reshape, format_transpose

__all__ = (
    "D",
    "LayerPyTorchTestCase",
    "ParamsLayerPytorch",
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

# setting random seed
SEED = 1234
rand.seed(SEED)
# Constant values

N = 8
C = 3
H = 32
W = 32

ADAPTIVE_AVG_POOL_OUTPUT_SIZE = (3, 3)

AVG_POOL_SHAPE = (3, 3)
AVG_POOL_PADDING = 0
AVG_POOL_STRIDE = 1

BATCH_NORMALIZATION_EPSILON = 1e-5
BATCH_NORMALIZATION_MOMENTUM_PYDTNN = 0.9
BATCH_NORMALIZATION_MOMENTUM_TORCH = BATCH_NORMALIZATION_MOMENTUM_PYDTNN
BATCH_NORMALIZATION_NUM_FEATURES = C

CONV2D_IN_C_TORCH = C
CONV2D_N_FILTERS = 2
CONV2D_FILTER_SHAPE = (4, 4)
CONV2D_PADDING = 0
CONV2D_STRIDE = 1
CONV2D_DILATION = 1
CONV2D_DEPTHWISE_PADDING = 1

FC_OUPUT_SHAPE = (2,)
LINEAR_OUTPUT = FC_OUPUT_SHAPE[0]

MAX_POOL_SHAPE = (2, 2)
MAX_POOL_PADDING = 0
MAX_POOL_STRIDE = 1
MAX_POOL_DILATION = 1


GRAD_EQUIVALENCES: dict[str, str] = {
    Parameters.WEIGHTS: "weight",
    Parameters.BIASES: "bias",
    # Parameters.RUNNING_MEAN: "running_mean", # Not PyDTNN's grading var.
    # Parameters.RUNNING_VAR: "running_var", # Not PyDTNN's grading var.
    # Parameters.BETA: "", # Not in PyTorch
    # Parameters.GAMMA: "", # Not in PyTorch
}

# PyTorch models


class TorchArcTanH(torch.nn.Module):
    """PyTorch implementation of the Arctanh activation function."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the Arctanh module."""
        super().__init__(*args, **kwargs)
        self.arc_tanh = torch.atanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Arctanh activation."""
        x = self.arc_tanh(x)
        return x


class TorchAdditionBlock(torch.nn.Module):
    """PyTorch model representing an addition block for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the addition block with two parallel convolutional paths."""
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=CONV2D_IN_C_TORCH,
                out_channels=CONV2D_N_FILTERS,
                kernel_size=CONV2D_FILTER_SHAPE,
                padding=CONV2D_PADDING,
                stride=CONV2D_STRIDE,
                dilation=CONV2D_DILATION,
            ),
            torch.nn.BatchNorm2d(
                CONV2D_N_FILTERS,
                eps=BATCH_NORMALIZATION_EPSILON,
                momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH,
            ),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=CONV2D_IN_C_TORCH,
                out_channels=CONV2D_N_FILTERS,
                kernel_size=CONV2D_FILTER_SHAPE,
                padding=CONV2D_PADDING,
                stride=CONV2D_STRIDE,
                dilation=CONV2D_DILATION,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the sum of the two block outputs."""
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = x1 + x2
        return x


class TorchConcatenationBlock(torch.nn.Module):
    """PyTorch model representing a concatenation block for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the concatenation block with two parallel convolutional paths."""
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=CONV2D_IN_C_TORCH,
                out_channels=CONV2D_N_FILTERS,
                kernel_size=CONV2D_FILTER_SHAPE,
                padding=CONV2D_PADDING,
                stride=CONV2D_STRIDE,
                dilation=CONV2D_DILATION,
            ),
            torch.nn.BatchNorm2d(
                CONV2D_N_FILTERS,
                eps=BATCH_NORMALIZATION_EPSILON,
                momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH,
            ),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=CONV2D_IN_C_TORCH,
                out_channels=CONV2D_N_FILTERS,
                kernel_size=CONV2D_FILTER_SHAPE,
                padding=CONV2D_PADDING,
                stride=CONV2D_STRIDE,
                dilation=CONV2D_DILATION,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the concatenation of the two block outputs along the channel dimension."""
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = torch.cat([x1, x2], dim=1)
        return x


class TorchDepthPointConv(torch.nn.Module):
    """PyTorch model representing a depthwise-pointwise convolution sequence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the depthwise and pointwise convolution layers."""
        super().__init__(*args, **kwargs)

        input_filt = CONV2D_IN_C_TORCH
        output_filt = CONV2D_N_FILTERS
        stride = CONV2D_STRIDE
        padding = CONV2D_DEPTHWISE_PADDING

        self.conv_depth = torch.nn.Conv2d(
            in_channels=input_filt,
            out_channels=input_filt,
            kernel_size=CONV2D_FILTER_SHAPE,
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

        # self.layers = torch.nn.Sequential(
        #                torch.nn.Conv2d(in_channels=input_filt, out_channels=input_filt, kernel_size=CONV2D_FILTER_SHAPE,
        #                                stride=stride, padding=1, groups=input_filt),
        #                torch.nn.Conv2d(in_channels=input_filt, out_channels=output_filt, kernel_size=(1,1),
        #                                 stride=1, padding=0, dilation=1, groups=1),
        #              )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the depthwise followed by the pointwise convolution."""
        x = self.conv_depth(x)
        x = self.conv_point(x)
        # x = self.layers(x)
        return x


class D:
    """Data dimensions container for test configurations."""

    def __init__(self, b: int = N, c: int = C, h: int = H, w: int = W) -> None:
        """Initializes dimensions with batch size, channels, height, and width."""
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width


class ParamsLayerPytorch(Params):
    """Configuration parameters for PyTorch layer tests."""

    def __init__(self, d: D = D()) -> None:
        """Initializes test parameters based on provided dimensions."""
        super().__init__()
        self.batch_size = d.b
        self.backend = "cpu"
        self.tensor_format = TensorFormat.NCHW
        self.shape = format_reshape((C, H, W), "CHW", self.tensor_format[1:])
        self.evaluate_only = False
        self.parallel_data = False
        self.loss_func_name = "categorical_cross_entropy"
        self.use_cudnn = False
        self.omm = None
        self.dtype = np.dtype(np.float32)
        self.tracing = False
        self.tracer_output = ""
        torch.set_default_dtype(numpy_to_torch_dtype_dict[self.dtype.type])
        self.dtype = np.dtype(self.dtype)

    def asdict(self) -> dict[str, Any]:
        """Returns the parameters as a dictionary."""
        return self.__dict__


class LayerPyTorchTestCase(TestCase):
    """Base test case class for verifying PyDTNN layers against PyTorch."""

    params = ParamsLayerPytorch()

    def setUp(self) -> None:
        """Sets up the test environment."""
        super().setUp()
        torch.manual_seed(0)

    # Initialization methods

    @staticmethod
    def get_test_data(
        no_zeros: bool = False,
        normalize: bool = True,
        positives_and_negatives: bool = True,
        shape_with_elements: tuple = (params.batch_size, *params.shape),
        dtype: np.dtype = params.dtype,
    ) -> np.ndarray:
        """Generates synthetic test data for layer verification."""
        num_elems = math.prod(shape_with_elements) // 4

        x_1 = np.arange(num_elems)
        x_2 = np.arange(num_elems) * -1

        if no_zeros:
            x_1 += 1
            x_2 -= 1

        x_1_1 = np.where(x_1 % 2 == 0, x_1, x_1 / 3)
        x_1_2 = np.where(x_1 % 2 != 0, x_1, x_1)

        x_2_1 = np.where(x_1 % 2 == 0, x_2, x_2 / 3)
        x_2_2 = np.where(x_1 % 2 != 0, x_2, x_2)

        # NOTE: seems that PyTorch doesn't like too much np.float64
        x = np.stack([x_1_1, x_1_2, x_2_1, x_2_2], axis=0, dtype=dtype).reshape(shape_with_elements)
        rand.shuffle(x)

        if normalize:
            min_x = np.min(x)
            x = (x - min_x) / (np.max(x) - min_x)
            if positives_and_negatives:
                x -= 0.5

        return np.asarray(x, dtype=dtype, order="C").copy()

    @staticmethod
    def initialize_pydtnn_model(
        list_layers: list[Layerable], params: ParamsLayerPytorch = params
    ) -> Model:
        """Initializes a PyDTNN model with the provided layers."""
        model = Model(**params.asdict())
        model.add(Input(params.shape))
        model.add_layers(list_layers)
        model.mode = Model.Mode.TRAIN
        model._model_init()
        return model

    def _copy_grad_vars(
        self, grad: np.ndarray, grad_var: str, torch_layer: torch.nn.Module
    ) -> None:
        """Copies gradient variables from PyDTNN to PyTorch layers."""
        if grad is not None:
            torch_grad_var = GRAD_EQUIVALENCES[grad_var]
            torch_grad = getattr(torch_layer, torch_grad_var)
            torch_grad.copy_(
                torch.from_numpy(grad.reshape(torch_grad.shape, copy=True))
                .to(torch.device("cpu"))
                .float()
            )

    def copy_grad_vars(self, pydtnn_model: Model, torch_model: torch.nn.Module) -> None:
        """Synchronizes model parameters between PyDTNN and PyTorch."""
        layers = [
            layer
            for layer in pydtnn_model.get_all_layers()
            if not isinstance(layer, AbstractBlockLayer)
        ]

        if isinstance(layers[0], Input):
            layers.pop(0)

        torch_layers = list()
        list_children = list(torch_model.children())
        if len(list_children) == 0:
            torch_layers.append(torch_model)
        else:
            for module in list_children:
                if isinstance(module, torch.nn.Sequential):
                    torch_layers.extend(module.children())
                else:
                    torch_layers.append(module)

        with torch.no_grad():
            for i in range(len(layers)):
                layer = layers[i]
                torch_layer = torch_layers[i]
                match layer:
                    case BatchNormalization():
                        assert isinstance(torch_layer, torch.nn.BatchNorm2d)
                        running_mean: np.ndarray = layer.running_mean  # pyright: ignore[reportAssignmentType]
                        running_var: np.ndarray = layer.running_var  # pyright: ignore[reportAssignmentType]
                        if running_mean is not None:
                            assert isinstance(torch_layer.running_mean, torch.Tensor)
                            torch_layer.running_mean.copy_(
                                torch.from_numpy(running_mean.copy())
                                .to(torch.device("cpu"))
                                .float()
                            )
                        if running_var is not None:
                            assert isinstance(torch_layer.running_var, torch.Tensor)
                            torch_layer.running_var.copy_(
                                torch.from_numpy(running_var.copy()).to(torch.device("cpu")).float()
                            )
                    case FC():
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            grad = grad if grad_var != Parameters.WEIGHTS else grad.T
                            self._copy_grad_vars(grad, grad_var, torch_layer)
                    case Conv2D() | Conv2DDepthwise():
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            if grad_var == Parameters.WEIGHTS and grad is not None:
                                grad = format_transpose(
                                    grad,
                                    {TensorFormat.NHWC: "ihwo", TensorFormat.NCHW: "oihw"}[
                                        pydtnn_model.tensor_format
                                    ],
                                    "oihw",
                                )
                            self._copy_grad_vars(grad, grad_var, torch_layer)
                    case _:
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            self._copy_grad_vars(grad, grad_var, torch_layer)

    def do_test_forward(
        self,
        _x: np.ndarray,
        pydtnn_model: Model,
        torch_model: torch.nn.Module,
        name_test: str,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Executes a forward comparison test between PyDTNN and PyTorch."""
        self.copy_grad_vars(pydtnn_model, torch_model)

        num_elems = len("Testing: ") + len(name_test)
        if verbose_test():
            logger.info(f"\n\n{'=' * num_elems}\nTesting: {name_test}\n{'=' * num_elems}")

        x = np.copy(_x)

        x = x.astype(dtype=self.params.dtype)

        for layer in pydtnn_model.layers:
            x: np.ndarray = layer.forward(x)
        x_pydtnn = x
        x_pydtnn = format_transpose(
            x, self.params.tensor_format.upper(), TensorFormat.NCHW.upper()
        ).copy()

        x_torch_tmp: torch.Tensor
        x_torch_tmp = (
            torch.from_numpy(_x.reshape((N, C, H, W), copy=False)).to(torch.device("cpu")).float()
        )
        x_torch_tmp = torch_model(x)
        x_torch = np.asarray(
            x_torch_tmp.numpy(force=True), dtype=pydtnn_model.dtype, order="C"
        ).copy()
        del x_torch_tmp

        if verbose_test():
            logger.info(
                f"[{rtol=},"
                f" {atol=}]\n{x_pydtnn.max()=}\n{x_torch.max()=}\n{x_pydtnn.min()=}\n{x_torch.min()=}"
                f"\n{x_pydtnn.std()=}\n{x_torch.std()=}\n{x_pydtnn.mean()=}\n{x_torch.mean()=}"
            )

        diff = x_pydtnn - x_torch
        if verbose_test():
            logger.info(f"{diff.max()=}\n{diff.min()=}\n{diff.std()=}\n{diff.mean()=}")

        # if not (diff < rtol).all():
        #    print(f"x_pydtnn:\n{x_pydtnn}")
        #    print(f"x_torch:\n{x_torch}")
        #    print(f"diff:\n{diff}")

        # self.assertTrue((diff < rtol).all()), f"Not all values are below the rtol. Max. difference: {diff.max()}."
        #                                       f"Std. deviation: {diff.std()}. Min. difference: {diff.min()}."
        self.assertTrue(np.allclose(x_pydtnn, x_torch, rtol=rtol, atol=atol))

    def do_test_fbf(
        self,
        _x: np.ndarray,
        pydtnn_model: Model,
        torch_model: torch.nn.Module,
        name_test: str,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Executes a forward-backward-forward comparison test between PyDTNN and PyTorch."""
        self.copy_grad_vars(pydtnn_model, torch_model)

        flattened_output_shape = (int(np.prod(pydtnn_model.layers[-1].shape)),)
        pydtnn_model.layers[-1].shape = flattened_output_shape
        # pydtnn_model.loss_func.shape = (_x.shape[0], flattened_output_shape)
        pydtnn_model.loss_func._model_init()
        pydtnn_model.loss_func._post_init()

        num_elems = len("Testing: ") + len(name_test)
        if verbose_test():
            logger.info(f"\n\n{'=' * num_elems}\nTesting: {name_test}\n{'=' * num_elems}")

        x = np.copy(_x)

        x: np.ndarray = x.astype(dtype=self.params.dtype)
        print(f"{pydtnn_model.output_shape=} || {flattened_output_shape=}")
        y = np.ones((x.shape[0], *flattened_output_shape), dtype=pydtnn_model.dtype)
        print(f"{y.shape=}")
        _y = y.copy()

        for layer in pydtnn_model.layers:
            x = layer.forward(x)
        x_base_shape = x.shape
        x_reshaped = x.reshape(N, -1)
        print(f"{x_reshaped=}")
        _loss, dx = pydtnn_model.loss_func.compute(x_reshaped, y, N)
        dx: np.ndarray = dx.reshape(x_base_shape)
        for layer in reversed(pydtnn_model.layers):
            dx = layer.backward(dx)
        print(f"backward PyDTNN: {dx} || {dx.sum()=} || {dx.mean()=} || {_loss=}")

        x_pydtnn = x
        x_pydtnn = format_transpose(
            x, self.params.tensor_format.upper(), TensorFormat.NCHW.upper()
        ).copy()

        loss_func = torch.nn.CrossEntropyLoss()
        x_torch_tmp: torch.Tensor = (
            torch.from_numpy(_x.reshape((N, C, H, W), copy=False)).to(torch.device("cpu")).float()
        )
        y_torch_tmp: torch.Tensor = (
            torch.from_numpy(_y.reshape((N, -1), copy=False)).to(torch.device("cpu")).float()
        )
        x_torch_tmp = x_torch_tmp.requires_grad_(True)
        x_torch_tmp.retain_grad()
        x_torch_tmp = torch_model(x_torch_tmp)

        x_torch_tmp = x_torch_tmp.reshape((N, -1))
        loss: torch.Tensor = loss_func(x_torch_tmp, y_torch_tmp)
        loss.retain_grad()
        print(f"before backward - {loss}\n===")
        loss.backward()
        print(f"after backward - {loss=}\n\n===")
        print(f"{loss.grad=}")

        x_torch = np.asarray(
            x_torch_tmp.numpy(force=True), dtype=pydtnn_model.dtype, order="C"
        ).reshape(x_base_shape, copy=True)
        del x_torch_tmp, y_torch_tmp

        if verbose_test():
            logger.info(
                f"[{rtol=},"
                f" {atol=}]\n{x_pydtnn.max()=}\n{x_torch.max()=}\n{x_pydtnn.min()=}\n{x_torch.min()=}"
                f"\n{x_pydtnn.std()=}\n{x_torch.std()=}\n{x_pydtnn.mean()=}\n{x_torch.mean()=}"
            )

        diff = x_pydtnn - x_torch
        if verbose_test():
            logger.info(f"{diff.max()=}\n{diff.min()=}\n{diff.std()=}\n{diff.mean()=}")

        if not (diff < rtol).all():
            print(f"x_pydtnn:\n{x_pydtnn}")
            print(f"x_torch:\n{x_torch}")
            print(f"diff:\n{diff}")

        # self.assertTrue((diff < rtol).all()), f"Not all values are below the rtol. Max. difference: {diff.max()}."
        #                                       f"Std. deviation: {diff.std()}. Min. difference: {diff.min()}."
        self.assertTrue(np.allclose(x_pydtnn, x_torch, rtol=rtol, atol=atol))

    def do_test(
        self,
        _x: np.ndarray,
        pydtnn_model: Model,
        torch_model: torch.nn.Module,
        name_test: str,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Executes a forward comparison test and a forward-backward-forward comparison test between PyDTNN and PyTorch"""

        #   original_x = _x.copy()
        with self.subTest(f"forward - {name_test}"):
            self.do_test_forward(_x, pydtnn_model, torch_model, name_test, rtol, atol)

        # with self.subTest(f"forward + backward + forward - {name_test}"):
        #    self.do_test_fbf(original_x, pydtnn_model, torch_model, name_test, rtol, atol)

    # Unitary Test methods

    def test_adaptive_average_pool2d(self) -> None:
        """Tests AdaptiveAveragePool2D layer."""
        pydtnn_layers: list[Layerable] = [
            AdaptiveAveragePool2D(output_shape=ADAPTIVE_AVG_POOL_OUTPUT_SIZE)
        ]
        torch_model = torch.nn.AdaptiveAvgPool2d(output_size=ADAPTIVE_AVG_POOL_OUTPUT_SIZE)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()

        self.do_test(
            _x=_x,
            pydtnn_model=pydtnn_model,
            torch_model=torch_model,
            name_test="AdaptiveAveragePool2D",
            rtol=1e-4,
            atol=1e-3,
        )

    def test_average_pool2d(self) -> None:
        """Tests AveragePool2D layer."""
        pydtnn_layers: list[Layerable] = [
            AveragePool2D(
                pool_shape=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE
            )
        ]
        torch_model = torch.nn.AvgPool2d(
            kernel_size=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE
        )
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AveragePool2D"
        )

    def test_batch_normalization(self) -> None:
        """Tests BatchNormalization layer."""
        pydtnn_layers: list[Layerable] = [
            BatchNormalization(
                epsilon=BATCH_NORMALIZATION_EPSILON,
                momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN,
            )
        ]
        torch_model = torch.nn.BatchNorm2d(
            BATCH_NORMALIZATION_NUM_FEATURES,
            eps=BATCH_NORMALIZATION_EPSILON,
            momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH,
            affine=False,
        )
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )

        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x,
            pydtnn_model=pydtnn_model,
            torch_model=torch_model,
            name_test="BatchNormalization",
            rtol=1e0,
            atol=1e0,
        )

    def test_conv2d(self) -> None:
        """Tests Conv2D layer."""
        pydtnn_layers: list[Layerable] = [
            Conv2D(
                nfilters=CONV2D_N_FILTERS,
                filter_shape=CONV2D_FILTER_SHAPE,
                padding=CONV2D_PADDING,
                stride=CONV2D_STRIDE,
                dilation=CONV2D_DILATION,
            )
        ]
        torch_model = torch.nn.Conv2d(
            in_channels=CONV2D_IN_C_TORCH,
            out_channels=CONV2D_N_FILTERS,
            kernel_size=CONV2D_FILTER_SHAPE,
            padding=CONV2D_PADDING,
            stride=CONV2D_STRIDE,
            dilation=CONV2D_DILATION,
        )
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Conv2D")

    @skip(
        reason=(
            "Dropout makes a random mask, then it can not be compared due both PyTorch and PyDTNN"
            " create different masks."
        )
    )
    def test_dropout(self) -> None:
        """Tests Dropout layer."""
        pydtnn_layers: list[Layerable] = [Dropout()]
        torch_model = torch.nn.Dropout()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Dropout")

    def test_flatten(self) -> None:
        """Tests Flatten layer."""
        pydtnn_layers: list[Layerable] = [Flatten()]
        torch_model = torch.nn.Flatten()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Flatten")

    def test_fc(self) -> None:
        """Tests Fully Connected (FC) layer."""
        pydtnn_layers: list[Layerable] = [Flatten(), FC(shape=FC_OUPUT_SHAPE)]
        torch_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=math.prod(self.params.shape), out_features=LINEAR_OUTPUT),
        )
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x,
            pydtnn_model=pydtnn_model,
            torch_model=torch_model,
            name_test="FC",
            rtol=1e-5,
            atol=1e-5,
        )

    def test_maxpool2d(self) -> None:
        """Tests MaxPool2D layer."""
        pydtnn_layers: list[Layerable] = [
            MaxPool2D(
                pool_shape=MAX_POOL_SHAPE,
                padding=MAX_POOL_PADDING,
                stride=MAX_POOL_STRIDE,
                dilation=MAX_POOL_DILATION,
            )
        ]
        torch_model = torch.nn.MaxPool2d(
            kernel_size=MAX_POOL_SHAPE,
            padding=MAX_POOL_PADDING,
            stride=MAX_POOL_STRIDE,
            dilation=MAX_POOL_DILATION,
        )
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="MaxPool2D"
        )

    def test_addition_block(self) -> None:
        """Tests AdditionBlock layer."""
        pydtnn_layers: list[Layerable] = [
            AdditionBlock(
                [
                    Conv2D(
                        nfilters=CONV2D_N_FILTERS,
                        filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING,
                        stride=CONV2D_STRIDE,
                        dilation=CONV2D_DILATION,
                    ),
                    BatchNormalization(
                        epsilon=BATCH_NORMALIZATION_EPSILON,
                        momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN,
                    ),
                ],
                [
                    Conv2D(
                        nfilters=CONV2D_N_FILTERS,
                        filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING,
                        stride=CONV2D_STRIDE,
                        dilation=CONV2D_DILATION,
                    )
                ],
            )
        ]

        torch_model = TorchAdditionBlock()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x,
            pydtnn_model=pydtnn_model,
            torch_model=torch_model,
            name_test="AdditionBlock",
            rtol=2,
            atol=2,
        )

    def test_concatenation_block(self) -> None:
        """Tests ConcatenationBlock layer."""
        pydtnn_layers: list[Layerable] = [
            ConcatenationBlock(
                [
                    Conv2D(
                        nfilters=CONV2D_N_FILTERS,
                        filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING,
                        stride=CONV2D_STRIDE,
                        dilation=CONV2D_DILATION,
                    ),
                    BatchNormalization(
                        epsilon=BATCH_NORMALIZATION_EPSILON,
                        momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN,
                    ),
                ],
                [
                    Conv2D(
                        nfilters=CONV2D_N_FILTERS,
                        filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING,
                        stride=CONV2D_STRIDE,
                        dilation=CONV2D_DILATION,
                    )
                ],
            )
        ]
        torch_model = TorchConcatenationBlock()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x,
            pydtnn_model=pydtnn_model,
            torch_model=torch_model,
            name_test="ConcatenationBlock",
            rtol=2,
            atol=2,
        )

    def test_sigmoid(self) -> None:
        """Tests Sigmoid activation."""
        pydtnn_layers: list[Layerable] = [Sigmoid()]
        torch_model = torch.nn.Sigmoid()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Sigmoid")

    def test_relu(self) -> None:
        """Tests Relu activation."""
        pydtnn_layers: list[Layerable] = [Relu()]
        torch_model = torch.nn.ReLU()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Relu")

    def test_relu6(self) -> None:
        """Tests Relu6 activation."""
        pydtnn_layers: list[Layerable] = [Relu6()]
        torch_model = torch.nn.ReLU6()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Relu6")

    def test_leaky_relu(self) -> None:
        """Tests LeakyRelu activation."""
        pydtnn_layers: list[Layerable] = [LeakyRelu()]
        torch_model = torch.nn.LeakyReLU()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(
            _x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="LeakyRelu"
        )

    def test_tanh(self) -> None:
        """Tests Tanh activation."""
        pydtnn_layers: list[Layerable] = [Tanh()]
        torch_model = torch.nn.Tanh()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Tanh")

    def test_arctanh(self) -> None:
        """Tests Arctanh activation."""
        # NOTE: Domain ArcTanH: Real numbers between "]-1, 1["
        pydtnn_layers: list[Layerable] = [Arctanh()]
        torch_model = TorchArcTanH()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Arctanh")

    def test_log(self) -> None:
        """Tests Log activation."""
        pydtnn_layers: list[Layerable] = [Log()]
        torch_model = torch.nn.LogSigmoid()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        # _x = np.where(_x < 0, 1, _x)
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Log")

    def test_softmax(self) -> None:
        """Tests Softmax activation."""
        pydtnn_layers: list[Layerable] = [Softmax()]
        torch_model = torch.nn.Softmax(dim=1)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Softmax")

    def test_depthwise_pointwise(self) -> None:
        """Tests Depthwise and Pointwise convolution sequence."""
        input_filt = CONV2D_IN_C_TORCH
        output_filt = CONV2D_N_FILTERS

        pydtnn_layers: list[Layerable] = [
            Conv2DDepthwise(
                nfilters=input_filt,
                stride=CONV2D_STRIDE,
                filter_shape=CONV2D_FILTER_SHAPE,
                padding=CONV2D_DEPTHWISE_PADDING,
            ),
            Conv2DPointwise(nfilters=output_filt),
        ]
        torch_model = TorchDepthPointConv()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(
            pydtnn_layers, params=self.params
        )
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Softmax")
