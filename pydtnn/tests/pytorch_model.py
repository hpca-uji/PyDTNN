"""Tests for verifying model behavior and consistency across different data types."""

import logging
import unittest

import numpy as np
import torch
import torchvision.models as torch_models
from pydtnn.tests.abstract.pytorch import ParamsTestTorchAbstract, PytorchModelTestCaseAbstract
from pydtnn.utils.tensor import TensorFormat

type PyTorch_Model = torch.nn.Module

__all__ = ("PytorchModelTestCase",)

logger = logging.getLogger(__name__)


class ResNet14Like(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1_0 = (
            torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
        )
        self.layer1_1 = (
            torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                256, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
            torch.nn.ReLU(inplace=True),
        )
        self.layer2_0 = (
            torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
        )
        self.layer3_0 = (
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
        self.layer3_1 = (
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
        self.layer4_0 = (
            torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                512, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
        )
        self.layer4_1 = (
            torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            torch.nn.BatchNorm2d(
                2048, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
            ),
        )

        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.layer1 = torch.nn.Sequential(*self.layer1_0, *self.layer1_1)
        self.layer2 = torch.nn.Sequential(*self.layer2_0)
        self.layer3 = torch.nn.Sequential(*self.layer3_0, *self.layer3_1)
        self.layer4 = torch.nn.Sequential(*self.layer4_0, *self.layer4_1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu2(x)
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
            # torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TorchLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x


class ParamsTestTorch(ParamsTestTorchAbstract):
    """Class where all the test parameters are stored"""

    def __init__(self) -> None:
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


class PytorchModelTestCase(PytorchModelTestCaseAbstract):
    """Tests that a PyTorch model and a PyDTNN model have to the same results."""

    # NOTE: Delete parent test to prevent re-export and re-testing
    global PytorchModelTestCaseAbstract
    del PytorchModelTestCaseAbstract

    def do_test_model(
        self,
        model_torch: PyTorch_Model,
        model_name: str,
        without_weighted_layers: bool = False,
    ) -> None:
        """Method to call the tests."""
        model_pydtnn = self.get_model_pydtnn(model_torch)
        super().do_test_model(model_torch, model_name, model_pydtnn, without_weighted_layers)

    @staticmethod
    def get_model_torch(model_name: str) -> PyTorch_Model:
        """
        Initializes a model and its corresponding loss function.

        Args:
            model_name: Name of the model to initialize.
            overwrite_params: Optional dictionary to override default parameters.

        Returns:
            A tuple containing the initialized PyTorch_Model and the list where the forward's outputs will be stored.
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
                torch_model = torch_models.resnet50(
                    weights=torch_models.ResNet50_Weights.IMAGENET1K_V1
                )
                torch_model.fc = torch.nn.Sequential(  # pyright: ignore[reportAttributeAccessIssue]
                    # torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(
                        in_features=torch_model.fc.in_features,
                        out_features=params.synthetic_output_shape[0],
                    )
                )
            case _:
                raise ValueError(f"Unknown model {model_name!r}!")

        PytorchModelTestCase.replace_layer_pytorch(torch_model, layer_to_replace=torch.nn.Dropout)
        PytorchModelTestCase.remove_inplace_pytorch(torch_model)

        return torch_model

    @unittest.skip("Large model")
    def test_renset50(self) -> None:
        """Compares results between an ResNet50 model using a PyTorch model and other a PyDTNN one."""
        # NOTE: It doesn't work well in GPU due an inplace operation (the hooks doesn't like that kind of operations).
        model_name = "resnet50"
        self.do_test_model(self.get_model_torch(model_name), model_name)

    # @unittest.skip("Work in progress.")
    def test_resnet14like(self) -> None:
        """Compares results between an ResNet14_like model using a PyTorch model and other a PyDTNN one."""
        model_name = "resnet14like"
        self.do_test_model(self.get_model_torch(model_name), model_name)

    # @unittest.skip("Work in progress.")
    def test_simplecnn(self) -> None:
        """Compares results between an SimpleCNN model using a PyTorch model and other a PyDTNN one."""
        model_name = "simplecnn"
        self.do_test_model(self.get_model_torch(model_name), model_name)

    # @unittest.skip("Work in progress.")
    def test_layer_conv_2d(self) -> None:
        """Compares results between an SimpleCNN model using a PyTorch model and other a PyDTNN one."""
        params = PytorchModelTestCase.params

        k_size = 3
        c_in = params.synthetic_input_shape[0]
        c_out = params.synthetic_output_shape[0]

        layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k_size),
            torch.nn.Flatten(),
        )

        torch_model = TorchLayer(layer)
        self.do_test_model(torch_model, "Conv2d")

    # @unittest.skip("Work in progress.")
    def test_layer_linear(self) -> None:
        """Compares results between an SimpleCNN model using a PyTorch model and other a PyDTNN one."""
        params = PytorchModelTestCase.params

        in_elems = int(np.prod(params.synthetic_input_shape))
        out_elems = int(np.prod(params.synthetic_output_shape))

        layer = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(in_features=in_elems, out_features=out_elems)
        )

        torch_model = TorchLayer(layer)
        self.do_test_model(torch_model, "Linear")

    # @unittest.skip("Work in progress.")
    def test_layer_batch_norm_2d(self) -> None:
        """Compares results between an SimpleCNN model using a PyTorch model and other a PyDTNN one."""
        params = PytorchModelTestCase.params

        eps = params.optimizer_epsilon
        momentum = params.optimizer_momentum
        c_in = params.synthetic_input_shape[0]

        layer = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=c_in, eps=eps, momentum=momentum), torch.nn.Flatten()
        )

        torch_model = TorchLayer(layer)
        self.do_test_model(torch_model, "BatchNorm2d")

    # @unittest.skip("Work in progress.")
    def test_maxpool2d(self) -> None:
        """Tests MaxPool2D layer."""
        # params = PytorchModelTestCase.params
        kernel_size = (3, 3)
        padding = 0
        stride = 1
        dilation = 1

        torch_model = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation
            ),
            torch.nn.Flatten(),
        )
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Max_Pool_2D", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_adaptive_average_pool2d(self) -> None:
        """Tests AdaptiveAveragePool2D layer."""
        params = PytorchModelTestCase.params
        output_size = params.synthetic_output_shape[0]

        torch_model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=output_size), torch.nn.Flatten()
        )
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Adaptive_Average_Pool", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_sigmoid(self) -> None:
        """Tests Sigmoid activation."""
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Sigmoid())
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Sigmoid", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_relu(self) -> None:
        """Tests Relu activation."""
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.ReLU())
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Relu", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_relu6(self) -> None:
        """Tests Relu6 activation."""
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.ReLU6())
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Relu6", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_leaky_relu(self) -> None:
        """Tests LeakyRelu activation."""
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LeakyReLU())
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "LeakyRelu", without_weighted_layers=True)

    @unittest.skip("Work in progress.")
    def test_tanh(self) -> None:
        """Tests Tanh activation."""
        # NOTE: It works well in GPU.
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Tanh())
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Tanh", without_weighted_layers=True)

    @unittest.skip("Work in progress.")
    def test_log_sigmoid(self) -> None:
        """Tests Log Sigmoid activation."""
        # NOTE: It works well in GPU.
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LogSigmoid())
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Log_Sigmoid", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_log_softmax(self) -> None:
        """Tests Log Softmax activation."""
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.LogSoftmax(dim=1))
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Log_Softmax", without_weighted_layers=True)

    # @unittest.skip("Work in progress.")
    def test_softmax(self) -> None:
        """Tests Softmax activation."""
        torch_model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Softmax(dim=1),
        )
        torch_model = TorchLayer(torch_model)
        self.do_test_model(torch_model, "Softmax", without_weighted_layers=True)
