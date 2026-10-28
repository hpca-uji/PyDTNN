"""Tests for verifying model behavior and consistency across different data types."""

import logging
import unittest
import warnings
from typing import Any

import numpy as np
import torch
from torch.optim import SGD, Adam, NAdam

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.converters.pytorch2pydtnn.model_converter import get_layers_from_torch
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.layers.abstract.layer import LayerError
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.model import Model as PyDTNN_Model
from pydtnn.model.base import ModelMode
from pydtnn.tests.abstract.base import Params, TestCase, verbose_test
from pydtnn.utils import header, rand
from pydtnn.utils.constants import Array, Parameters
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.tensor_array import TensorArray

type PyTorch_Model = torch.nn.Module

__all__ = ("PytorchModelTestCaseAbstract",)

logger = logging.getLogger(__name__)


class ParamsTestTorchAbstract(Params):
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

    @property
    def device(self) -> str:
        return self.backend if self.backend == "cpu" else "cuda"


class PytorchModelTestCaseAbstract(TestCase):
    """Tests that a PyTorch model and a PyDTNN model have to the same results."""

    # NOTE: Delete parent test to prevent re-export and re-testing
    global TestCase
    del TestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model2_desc}
    model1_desc = "using PyTorch"
    model2_desc = "using PyDTNN"

    rtol_default = 1e-4  # 1e-10  # 1e-4
    atol_default = 1e-4  # 1e-10  # 1e-4
    rtol_dict = {}  # BatchNormalization: 1e-5, Conv2D: 1e-5,
    atol_dict = {}  # BatchNormalization: 1e-4, Conv2D: 1e-5,
    params = ParamsTestTorchAbstract()

    def setUp(self) -> None:
        """Sets up the test environment."""
        super().setUp()
        torch.manual_seed(0)

    @staticmethod
    def replace_layer_pytorch(
        module: torch.nn.Module, layer_to_replace: type[torch.nn.Module]
    ) -> torch.nn.Module | None:
        """
        Recursively put desired batch norm in nn.module module.

        set module = net to start code.
        https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/7
        """

        # iterate through immediate child modules
        list_children = list(module.named_children())
        for name, immediate_child_module in list_children:
            setattr(module, name, PytorchModelTestCaseAbstract.replace_layer_pytorch(immediate_child_module, layer_to_replace))

        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        if isinstance(module, layer_to_replace):
            if verbose_test():
                print(f"Removed: {module}")
            return torch.nn.Identity()
        else:
            return module

    @staticmethod
    def remove_inplace_pytorch(torch_model: torch.nn.Module) -> None:
        """Recursively replaces inplace methods with normal methods."""

        def _get_all_layers_pytorch(module: torch.nn.Module) -> None:
            # iterate through immediate child modules
            list_children = list(module.named_children())
            for _name, immediate_child_module in list_children:
                if len(list(immediate_child_module.named_children())) == 0:
                    if hasattr(immediate_child_module, "inplace"):
                        if verbose_test():
                            print(
                                f"INPLACE (before) -> {immediate_child_module} || {immediate_child_module.inplace=} =="
                            )
                        immediate_child_module.inplace = False  # pyright: ignore[reportArgumentType] (It's fine)
                        if verbose_test():
                            print(
                                f"INPLACE (after) -> {immediate_child_module} || {immediate_child_module.inplace=} =="
                            )
                _get_all_layers_pytorch(immediate_child_module)

        _get_all_layers_pytorch(torch_model)

    @staticmethod
    def set_forward_hook(torch_model: torch.nn.Module) -> list[tuple[torch.nn.Module, torch.Tensor]]:
        """Recursively sets the forward's 'hook' method."""
        # params = PytorchModelTestCaseAbstract.params
        torch_forward_outputs = list[tuple[torch.nn.Module, torch.Tensor]]()

        def hook(
            module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor
        ) -> torch.Tensor | None:
            _output = output.clone()
            # if params.dtype is np.dtype(np.float64):
            #    _output = _output.double()
            # else:
            #    _output = _output.float()

            torch_forward_outputs.append((module, _output))
            # return output

        def _get_all_layers_pytorch(module: torch.nn.Module) -> None:
            # iterate through immediate child modules
            list_children = list(module.named_children())
            for _name, immediate_child_module in list_children:
                if len(list(immediate_child_module.named_children())) == 0:
                    immediate_child_module.register_forward_hook(hook)
                _get_all_layers_pytorch(immediate_child_module)

        _get_all_layers_pytorch(torch_model)

        return torch_forward_outputs

    @staticmethod
    def set_backward_hook(
        torch_model: torch.nn.Module,
    ) -> list[tuple[torch.nn.Module, tuple[torch.Tensor, ...] | torch.Tensor]]:
        """Recursively sets the backwards's 'hook' method."""

        # Forward
        # Input -> Layer -> Output
        # PyDTNN's Backward:
        # Output <- Layer <- Input
        # PyDTNN's Backward:
        # Input.grad <- Layer <- Output.grad

        # torch_backward_outputs = list[tuple[torch.nn.Module, tuple[torch.Tensor, ...] | torch.Tensor]]()
        torch_backward_inputs = list[tuple[torch.nn.Module, tuple[torch.Tensor, ...] | torch.Tensor]]()

        # params = PytorchModelTestCaseAbstract.params

        # _grad_t = tuple[Tensor, ...] | Tensor
        def hook(
            module: torch.nn.Module,
            grad_input: tuple[torch.Tensor, ...] | torch.Tensor,
            grad_output: tuple[torch.Tensor, ...] | torch.Tensor,
        ) -> tuple[torch.Tensor, ...] | torch.Tensor | None:
            """Torch backward's hook."""
            output_copy = tuple([elem if elem is None else elem.clone() for elem in grad_output])
            # output_copy = tuple([elem if elem is None else
            #                    (elem.clone().double() if params.dtype is np.dtype(np.float64) else elem.clone().float())
            #                    for elem in grad_output])
            # torch_backward_outputs.append((module, output_copy))

            input_copy = tuple([elem if elem is None else elem.clone() for elem in grad_input])

            # input_copy = tuple([elem if elem is None else
            #                    (elem.clone().double() if params.dtype is np.dtype(np.float64) else elem.clone().float())
            #                    for elem in grad_input])
            if verbose_test():
                print(f"{module} - {input_copy[0].size()=} - {output_copy[0].size()=}")
            torch_backward_inputs.append((module, input_copy))

        def _get_all_layers_pytorch(module: torch.nn.Module) -> None:
            # iterate through immediate child modules
            list_children = list(module.named_children())
            for _name, immediate_child_module in list_children:
                if len(list(immediate_child_module.named_children())) == 0:
                    immediate_child_module.register_full_backward_hook(hook)
                _get_all_layers_pytorch(immediate_child_module)

        _get_all_layers_pytorch(torch_model)

        return torch_backward_inputs

    @staticmethod
    def get_all_pytorch_layers(
        torch_model: torch.nn.Module,
    ) -> list[torch.nn.Module]:
        """Recursively gets all the layers of a PyTorch model."""

        layers = list[torch.nn.Module]()

        def _get_all_layers_pytorch(module: torch.nn.Module) -> None:
            # iterate through immediate child modules
            list_children = list(module.named_children())
            for _name, immediate_child_module in list_children:
                layers.append(immediate_child_module)
                _get_all_layers_pytorch(immediate_child_module)

        _get_all_layers_pytorch(torch_model)

        return layers

    @staticmethod
    def get_torch_grad_parameters_values(
        torch_model: PyTorch_Model,
    ) -> dict[int, dict[str, tuple[torch.nn.Module, np.ndarray | None]]]:
        """Function to get all the parameter's gradients."""
        layers = PytorchModelTestCaseAbstract.get_all_pytorch_layers(torch_model)
        # weight = weights (dw); bias = biases (db)
        equivalence = {"weight": Parameters.DW, "bias": Parameters.DB}
        layers_grad_vars = dict[int, dict[str, tuple[torch.nn.Module, np.ndarray | None]]]()

        i = 0
        for layer in layers:
            name = layer._get_name()
            match name:
                case "BatchNorm2d" | "Conv2d" | "Linear":
                    if verbose_test():
                        print(
                            f"number: {i}, {layer=}, layer.bias={layer.bias is not None}, "
                            f"parameters: {layer._parameters.keys()}"
                        )
                    parameters = dict[str, tuple[torch.nn.Module, np.ndarray | None]]()
                    for key in layer._parameters.keys():
                        param = layer._parameters[key]
                        if param is not None:
                            param = None if param.grad is None else param.grad.numpy(force=True)
                        parameters[equivalence[key]] = (layer, param)
                    layers_grad_vars[i] = parameters
                    i += 1
                case _:
                    pass
                    # if verbose_test():
                    #     print(f"number: {i}, layer: {name}, with no parameters: {layer._parameters}")
        return layers_grad_vars

    @staticmethod
    def get_pydtnn_grad_vars_values(
        pydtnn_model: PyDTNN_Model,
    ) -> dict[int, tuple[Layerable, dict[str, np.ndarray | None]]]:
        """Function to get all the parameter's gradients."""
        # layers = pydtnn_model.layers
        layers_grad_vars = dict[int, tuple[Layerable, dict[str, np.ndarray | None]]]()

        def _get_pydtnn_grad_vars_values(i: int, layers: list[Layerable]) -> int:
            for layer in layers:
                name = layer.name_with_id
                match layer:
                    case BatchNormalization() | Conv2D() | FC() | Conv2DDepthwise() | Conv2DPointwise():
                        if verbose_test():
                            print(
                                f"number: {i}, {name}, {layer.use_bias=}, grad_vars: {layer.grad_vars.keys()}"
                            )
                        grad_vars = dict[str, np.ndarray | None]()

                        for grad_v_key in layer.grad_vars.values():
                            pydtnn_grad = getattr(layer, grad_v_key)
                            if isinstance(pydtnn_grad, TensorArray):
                                pydtnn_grad = pydtnn_grad.get().reshape(pydtnn_grad.cpu_shape)
                            grad_vars[grad_v_key] = pydtnn_grad
                        layers_grad_vars[i] = (layer, grad_vars)
                        i += 1
                    case AbstractBlockLayer():
                        for path in layer.paths:
                            i = _get_pydtnn_grad_vars_values(i, path)
                    case _:
                        pass
                        # if verbose_test():
                        #     print(f"number: {i}, layer: {name}, with no grad_vars: {layer.grad_vars}")
            return i

        _get_pydtnn_grad_vars_values(0, pydtnn_model.layers)

        return layers_grad_vars

    @staticmethod
    def get_torch_parameters_values(
        torch_model: PyTorch_Model,
    ) -> dict[int, dict[str, np.ndarray | None]]:
        """Function to get all the parameter's gradients."""
        layers = PytorchModelTestCaseAbstract.get_all_pytorch_layers(torch_model)
        # weight = weights (dw); bias = biases (db)
        equivalence = {
            "weight": Parameters.WEIGHTS,
            "bias": Parameters.BIASES,
            "running_mean": Parameters.RUNNING_MEAN,
            "running_var": Parameters.RUNNING_VAR,
        }
        layers_parameters = dict[int, dict[str, np.ndarray | None]]()

        i = 0
        for layer in layers:
            name = layer._get_name()

            match name:
                case "BatchNorm2d" | "Conv2d" | "Linear":
                    if verbose_test():
                        print(
                            f"number: {i}, {layer=}, layer.bias={layer.bias is not None}, "
                            f"parameters: {layer._parameters.keys()}"
                        )
                    parameters = dict[str, np.ndarray | None]()

                    for key in layer._parameters.keys():
                        param = layer._parameters[key]
                        if param is not None:
                            param = None if param is None else param.numpy(force=True)
                        parameters[equivalence[key]] = param
                    layers_parameters[i] = parameters
                    i += 1
                case _:
                    if verbose_test():
                        print(f"number: {i}, layer: {name}, with no parameters: {layer._parameters}")
        return layers_parameters

    @staticmethod
    def get_pydtnn_vars_values(pydtnn_model: PyDTNN_Model) -> dict[int, dict[str, np.ndarray | None]]:
        """Function to get all the parameter's gradients."""
        # layers = pydtnn_model.get_all_layers()
        layers_parameters = dict[int, dict[str, np.ndarray | None]]()

        def _get_pydtnn_vars_values(i: int, layers: list[Layerable]) -> int:
            for layer in layers:
                name = layer.name_with_id

                match layer:
                    case BatchNormalization() | Conv2D() | FC() | Conv2DDepthwise() | Conv2DPointwise():
                        if verbose_test():
                            print(
                                f"number: {i}, {layer=}, {layer.use_bias=}, grad_vars: {layer.grad_vars.keys()}"
                            )
                        parameters = dict[str, np.ndarray | None]()
                        for param_key in layer.grad_vars.keys():
                            param = getattr(layer, param_key)
                            if isinstance(param, TensorArray):
                                param = param.get().reshape(param.cpu_shape)
                            parameters[param_key] = param
                        layers_parameters[i] = parameters
                        i += 1
                    case AbstractBlockLayer():
                        for path in layer.paths:
                            i = _get_pydtnn_vars_values(i, path)
                    case _:
                        if verbose_test():
                            print(f"number: {i}, layer: {name}, with no params: {layer.grad_vars}")
            return i

        _get_pydtnn_vars_values(0, pydtnn_model.layers)

        return layers_parameters

    @staticmethod
    def delete_torch_flatten_and_identity_outputs(
        torch_outputs: list[tuple[torch.nn.Module, Any]],
    ) -> None:
        """Function to remove the flatten and identity layer's outputs"""
        indexes_to_remove = list[int]()
        for i, (layer, _) in enumerate(torch_outputs):
            if isinstance(layer, (torch.nn.Flatten, torch.nn.Identity)):
                indexes_to_remove.append(i)
        for index in reversed(indexes_to_remove):
            del torch_outputs[index]

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
    def _get_torch_loss_func() -> torch.nn.modules.loss._Loss:
        # [NOTE] Remeber: "torch.nn.CrossEntropyLoss = LogSoftmax() + NegativeLLLoss"
        return torch.nn.CrossEntropyLoss()

    def get_optimizer_pytorch(self, model_torch: PyTorch_Model) -> torch.optim.Optimizer:
        """Method to get PyTorch's optimizer"""
        params = self.params
        match params.optimizer_name:
            case "sgd":
                optimizer = SGD(
                    model_torch.parameters(),
                    lr=params.learning_rate,
                    momentum=params.optimizer_momentum,
                    nesterov=params.optimizer_nesterov,
                    weight_decay=params.optimizer_decay,
                    dampening=0.0,
                )
            case "adam":
                betas = (params.optimizer_beta1, params.optimizer_beta2)
                optimizer = Adam(
                    model_torch.parameters(),
                    lr=params.learning_rate,
                    betas=betas,
                    eps=params.optimizer_epsilon,
                    weight_decay=params.optimizer_decay,
                    decoupled_weight_decay=params.optimizer_decoupled_decay,
                    amsgrad=False,
                )
            case "nadam":
                betas = (params.optimizer_beta1, params.optimizer_beta2)
                optimizer = NAdam(
                    model_torch.parameters(),
                    lr=params.learning_rate,
                    betas=betas,
                    eps=params.optimizer_epsilon,
                    weight_decay=params.optimizer_decay,
                )
            case _:
                optimizer = None
                raise NotImplementedError(
                    f"Not implemented this test for {params.optimizer_name} optimizer."
                )
        return optimizer

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
        params = self.params
        params.loss_name = "negative_likelihood"
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

        layers = get_layers_from_torch(model_pytorch, params.synthetic_input_shape, LogSoftmax())
        model_pydtnn.add_layers(layers)
        model_pydtnn._model_init()
        model_pydtnn.mode = ModelMode.TRAIN

        if verbose_test():
            for layer in model_pydtnn.layers:
                print(layer)

        # capa = model_pydtnn.layers[-1]
        model_pydtnn.mode = ModelMode.TRAIN
        return model_pydtnn

    def do_torch_model_forward_pass(
        self, torch_model: PyTorch_Model, x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass for the PyTorch model.

        Args:
            model1: The model instance.
            x0: Initial input list.

        Returns:
            torch.Tensor: The last layer's output.
            The list of layer's outputs must be set in the hook. The variable should be "torch_forward_outputs"
        """
        return torch_model(x0)

    def do_pydtnn_model_forward_pass(
        self, pydtnn_model: PyDTNN_Model, x0: Array
    ) -> tuple[list[Array], list[tuple[Layerable, Array]]]:
        """
        Performs a forward pass for the PyDTNN model.

        Args:
            model1: The model instance.
            x0: Initial input list.

        Returns:
            List of outputs after each layer.
        """
        if verbose_test():
            print("Start 'do_pydtnn_model_forward_pass'")

        x1 = list[Array]()
        x_to_compare = list[tuple[Layerable, Array]]()
        for layer in pydtnn_model.layers:
            x0_base = x0  # No hace falta copiarlo, se va a "machacar" el valor luego
            if verbose_test():
                print(f"{layer}")

            if isinstance(layer, AbstractBlockLayer):
                for path in layer.paths:
                    _x0: Array = x0_base.copy()
                    for _layer in path:
                        _x0 = _layer.forward(_x0.copy())  # pyright: ignore[reportAssignmentType]
                        if not isinstance(_layer, (Identity, Flatten, AbstractBlockLayer)):
                            x_to_compare.append((_layer, _x0.copy()))

            # Every layer:
            x0 = layer.forward(x0.copy())  # pyright: ignore[reportAssignmentType]
            x1.append(x0.copy())

            if not isinstance(layer, (Identity, Flatten, AbstractBlockLayer)):
                x_to_compare.append((layer, x0.copy()))

        if verbose_test():
            print("End 'do_pydtnn_model_forward_pass'")
        return x1, x_to_compare

    def do_pytorch_model_loss(
        self, loss_func: torch.nn.modules.loss._Loss, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Method execute the pytorch's loss"""
        loss: torch.Tensor = loss_func(x, y)
        return loss

    def do_pydtnn_model_loss(self, model: PyDTNN_Model, x: Array, y: Array) -> tuple[float, Array]:
        """Method execute the pydtnn's loss"""
        model.real_batch_size = model.batch_size
        loss, dx = model.loss.compute(x.copy(), y)
        return loss, dx

    def compare_loss(self, loss_torch: torch.Tensor, _loss_pydtnn: float, round: int) -> None:
        """Method to compare Torch and PyDTNN losses."""
        _loss_torch = float(loss_torch.detach())
        if verbose_test():
            print(f"{_loss_torch=} || {_loss_pydtnn=}")
        rtol: float = self.rtol_default
        atol: float = self.atol_default
        assert np.isclose(float(_loss_torch), _loss_pydtnn, rtol=rtol, atol=atol), (
            f"[ROUND {round}] Both values are not close: {_loss_torch=} =/=  {_loss_pydtnn=}"
        )

    def do_pytorch_model_backward_pass(
        self, _torch_model: PyTorch_Model, loss: torch.Tensor
    ) -> None:
        """
        Performs a backward pass for PyTorch

        Args:
            _torch_model: The PyTorch model
            loss: the loss' function output

        Returns:
            Nothing special yet.
        """
        # dx: list[torch.Tensor] = []
        with warnings.catch_warnings(action="ignore"):
            loss.backward()

    def do_pydtnn_model_backward_pass(
        self, pydtnn_model: PyDTNN_Model, dx: Array
    ) -> list[tuple[Layerable, Array]]:
        """
        Performs a backward pass for PyDTNN's Model.

        Args:
            pydtnn_model: The model instance.
            dx: Initial gradient.

        Returns:
            List of gradients after each layer.
        """
        dx_to_compare = list[tuple[Layerable, Array]]()
        if verbose_test():
            print("Starting backward pass")

        for layer in reversed(pydtnn_model.layers):
            dx_base = dx
            if verbose_test():
                print(f"{layer} - {dx.shape}")

            if isinstance(layer, AbstractBlockLayer):
                # NOTE: Manually doing the AbstractBlockLayer backwards to get the intermediate outputs.
                paths_outputs = list[Array]()
                # for i in reversed(layer.paths):
                for i in range(len(layer.paths) - 1, -1, -1):
                    path = layer.paths[i]

                    if isinstance(layer, ConcatenationBlock):
                        # Splitting the concatenation
                        if self.params.device == "cuda":
                            # NOTE: First moving the values to CPU, doing the split, then moving the values back to GPU.
                            assert isinstance(dx_base, TensorArray)
                            dx_to_split: np.ndarray = dx_base.get().reshape(dx_base.cpu_shape)
                        else:
                            assert isinstance(dx_base, np.ndarray)
                            dx_to_split: np.ndarray = dx_base.copy()

                        dx_splitted = np.split(dx_to_split, layer.idx_co[:-1], axis=layer.concat_dim)[i]

                        if self.params.device == "cuda":
                            # NOTE: Moving the values back to GPU.
                            assert isinstance(dx_base, TensorArray)
                            _dx = TensorArray.to_gpu(dx_splitted, dx_base.tensor_format,
                                                     dx_base.cudnn_dtype, dx_base.tensor_type)
                        else:
                            _dx = dx_splitted
                    else:
                        _dx: Array = dx_base.copy()

                    for _layer in reversed(path):
                        _dx = _layer.backward(_dx.copy())  # pyright: ignore[reportAssignmentType]
                        if not isinstance(_layer, (Identity, Flatten, AbstractBlockLayer)):
                            dx_to_compare.append((_layer, _dx.copy()))
                    paths_outputs.append(_dx.copy())
                dx = np.sum(paths_outputs, axis=0)
            else:
                dx = layer.backward(dx.copy())
                if not isinstance(layer, (Identity, Flatten, AbstractBlockLayer)):
                    dx_to_compare.append((layer, dx))

        if verbose_test():
            print("Ending backward pass")

        return dx_to_compare

    def do_pytorch_model_optimizer_pass(
        self, _torch_model: PyTorch_Model, optimizer: torch.optim.Optimizer
    ) -> None:
        """
        Execute the PyTorch's optimizer.

        Args:
            _torch_model: The PyTorch model
            optimizer: The PyTorch optimizer.

        Returns:
            Nothing special yet.
        """
        optimizer.step()

    def do_pydtnn_model_optimizer_pass(self, pydtnn_model: PyDTNN_Model) -> None:
        """
        Execute the PyDTNN's optimizer.

        Args:
            model: The model instance.

        Returns:
            List of gradients after each layer.
        """
        for layer in pydtnn_model.layers:
            layer.update_weights(pydtnn_model.optimizer, update=True, sync=False)

    def _compare_fwd_bwd(
        self,
        layers_and_values_torch: list[tuple[torch.nn.Module, torch.Tensor]]
        | list[tuple[torch.nn.Module, tuple[torch.Tensor, ...] | torch.Tensor]],  # noqa: E501
        layers_and_values_pydtnn: list[tuple[Layerable, Array]],
        round: int,
        off_set_pydtnn: int = 0,
        message: str = "",
    ) -> None:
        """Method to compare both forward and backward's outputs."""

        if verbose_test():
            print(f"[Round: {round}] Comparing outputs of both models {message}...")
        self.delete_torch_flatten_and_identity_outputs(layers_and_values_torch)

        for i in range(len(layers_and_values_torch)):
            torch_layer, pytorch_base_values = layers_and_values_torch[i]
            pydtnn_layer, pydtnn_values = layers_and_values_pydtnn[i + off_set_pydtnn]
            if isinstance(pytorch_base_values, tuple):
                pytorch_base_values = pytorch_base_values[0]
            pytorch_values = pytorch_base_values.numpy(force=True)

            if self.params.device == "cuda":
                assert isinstance(pydtnn_values, TensorArray)
                pydtnn_values = pydtnn_values.get().reshape(pydtnn_values.cpu_shape)
            else:
                assert isinstance(pydtnn_values, np.ndarray)
            # try:
            rtol, atol = self.get_tolerance(pydtnn_layer)
            self.assertTrue(
                pytorch_values.size == pydtnn_values.size,
                f"[Round: {round}] Both tensors (PyDTNN: {pydtnn_layer.name_with_id}, PyTorch: {torch_layer})"
                f" doesn't have the same number of elements ({pytorch_values.size=} != {pydtnn_values.size=})",
            )
            self.assertTrue(
                np.allclose(
                    pytorch_values.flatten(), pydtnn_values.flatten(), rtol=rtol, atol=atol
                ),
                f"[Round: {round}] {message} result from layers {pydtnn_layer.name_with_id} and "
                f" {torch_layer} differ [layer: {i}] ({self.print_stats(pytorch_values, pydtnn_values, rtol, atol)})",
            )
            # except Exception as e:
            #     print(e)
            #     breakpoint()

    def compare_forward(
        self,
        values_torch: list[tuple[torch.nn.Module, torch.Tensor]],
        values_pydtnn: list[tuple[Layerable, Array]],
        round: int,
    ) -> None:
        """
        Compares the forward pass outputs of two models.

        Args:
            values_torch: Forward pass outputs of PyTorch's model.
            values_pydtnn: Forward pass outputs of PyDTNN's model.
        """
        self._compare_fwd_bwd(values_torch, values_pydtnn, round=round, message="forwards")

    def compare_backward(
        self,
        values_torch: list[tuple[torch.nn.Module, tuple[torch.Tensor, ...] | torch.Tensor]],
        values_pydtnn: list[tuple[Layerable, Array]],
        round: int,
        off_set_pydtnn: int = 1
    ) -> None:
        """Compares the backward pass gradients of two models.

        Due the differences of implementation, right now it's not possible to compare the both gradients.

        Args:
            values_torch: Backward pass gradients of torch's model.
            values_pydtnn: Backward pass gradients of pydtnn's model.
        """
        # NOTE: The offset is 1 to ignore the last pydtnn's activation.
        self._compare_fwd_bwd(
            values_torch, values_pydtnn, round=round, off_set_pydtnn=off_set_pydtnn, message="backwards"
        )

    def _compare_grad_vars(
        self,
        torch_gradients: dict[int, dict[str, tuple[torch.nn.Module, np.ndarray | None]]],
        pydtnn_gradients: dict[int, tuple[Layerable, dict[str, np.ndarray | None]]],
        round: int
    ) -> None:
        assert len(torch_gradients) == len(pydtnn_gradients), (
            f"[ROUND {round}] Torch and PyDTNN gradients "
            "must have the same number of elements: "
            f"{len(torch_gradients)=} || {len(pydtnn_gradients)=}"
        )
        for i in pydtnn_gradients.keys():
            pydtnn_layer, pydtnn_grad_vars = pydtnn_gradients[i]
            for grad_var_k in pydtnn_grad_vars.keys():
                torch_layer, torch_grad = torch_gradients[i][grad_var_k]
                pydtnn_grad = pydtnn_grad_vars[grad_var_k]
                if verbose_test():
                    print(f"Comparing {pydtnn_layer=} and {torch_layer=}'s '{grad_var_k}'.")

                if torch_grad is None:
                    assert pydtnn_grad is None, (
                        f"[ROUND {round}] {torch_layer} torch's gradient is None, "
                        f"but the PyDTNN {pydtnn_layer} is: {pydtnn_grad}"
                    )
                else:
                    assert pydtnn_grad is not None, (
                        f"[ROUND {round}] {torch_layer} torch's gradient is not None, "
                        f"but the PyDTNN {pydtnn_layer} it is."
                    )
                    assert torch_grad.size == pydtnn_grad.size, (
                        f"[ROUND {round}] {pydtnn_layer} Both tensors must "
                        "have the same size: "
                        f"({torch_grad.size=} =/= {pydtnn_grad.size=})"
                    )

                    if (
                        torch_grad.shape != pydtnn_grad.shape
                        and torch_grad.T.shape == pydtnn_grad.shape
                    ):
                        if verbose_test():
                            print(
                                f"[ROUND {round}] The layer's values are transposed on this layer "
                                f"({pydtnn_layer.name_with_id}). ({torch_grad.shape=} != {pydtnn_grad.shape=})."
                            )
                        torch_grad = torch_grad.T

                    rtol, atol = self.get_tolerance(pydtnn_layer)
                    assert np.isclose(
                        torch_grad.flatten(), pydtnn_grad.flatten(), rtol=rtol, atol=atol
                    ).all(), (
                        f"[ROUND {round}] Both values of {pydtnn_layer.name_with_id}'s {grad_var_k} are not close enough: "
                        f"{torch_grad.flatten()=}\n{pydtnn_grad.flatten()=}"
                        f"({self.print_stats(torch_grad, pydtnn_grad, rtol, atol)})"
                    )

    def compare_grad_vars(
        self, torch_model: PyTorch_Model, pydtnn_model: PyDTNN_Model, round: int
    ) -> None:
        """Method to comparte PyTorch 'parameters.grad' and PyDTNN grad vars's values"""

        if verbose_test():
            print(f"[Round: {round}] Comparing outputs of both models gradients...")

        torch_gradients = self.get_torch_grad_parameters_values(torch_model)
        pydtnn_gradients = self.get_pydtnn_grad_vars_values(pydtnn_model)
        self._compare_grad_vars(torch_gradients, pydtnn_gradients, round)

    def compare_optimizer_parameters(
        self, torch_model: PyTorch_Model, pydtnn_model: PyDTNN_Model, round: int
    ) -> None:
        """Method to comparte PyTorch 'parameters.grad' and PyDTNN grad vars's values"""

        if verbose_test():
            print(f"[Round: {round}] Comparing outputs of both models optimizer's parameters...")

        torch_params = self.get_torch_parameters_values(torch_model)
        pydtnn_params = self.get_pydtnn_vars_values(pydtnn_model)
        return self._compare_optimizer_parameters(torch_params, pydtnn_params, pydtnn_model, round)

    def _compare_optimizer_parameters(
        self,
        torch_params: dict[int, dict[str, np.ndarray | None]],
        pydtnn_params: dict[int, dict[str, np.ndarray | None]],
        pydtnn_model: PyDTNN_Model,
        round: int
    ) -> None:
        """The actual method to comparte PyTorch 'parameters.grad' and PyDTNN grad vars's values"""
        assert len(torch_params) == len(pydtnn_params), (
            f"[ROUND {round}] Torch and PyDTNN parameters must have the same "
            f"number of elements: {len(torch_params)=} || "
            f"{len(pydtnn_params)=}"
        )

        for i in pydtnn_params.keys():
            pydtnn_layer = pydtnn_model.get_all_layers()[i]
            for var_key in pydtnn_params[i].keys():
                torch_param = torch_params[i][var_key]
                pydtnn_param = pydtnn_params[i][var_key]
                if verbose_test():
                    print(f"Comparing {pydtnn_layer=}'s '{var_key}'.")

                if torch_param is None:
                    assert pydtnn_param is None, (
                        f"[ROUND {round}] {pydtnn_layer} torch's parameter is None, "
                        f"but the PyDTNN one is: {pydtnn_param}"
                    )
                else:
                    assert pydtnn_param is not None, (
                        f"[ROUND {round}] {pydtnn_layer} torch's parameter is not None, "
                        "but the PyDTNN it is."
                    )
                    assert torch_param.size == pydtnn_param.size, (
                        f"[ROUND {round}] {pydtnn_layer} Both tensors must have the "
                        f"same size: ({torch_param.size=} =/= {pydtnn_param.size=})"
                    )

                    if (
                        torch_param.shape != pydtnn_param.shape
                        and torch_param.T.shape == pydtnn_param.shape
                    ):
                        if verbose_test():
                            print(
                                f"[ROUND {round}] The layer's values are transposed on this layer "
                                f"({pydtnn_layer.name_with_id}). ({torch_param.shape=} != {pydtnn_param.shape=})."
                            )
                        torch_param = torch_param.T

                    rtol, atol = self.get_tolerance(pydtnn_layer)
                    assert np.isclose(
                        torch_param.flatten(), pydtnn_param.flatten(), rtol=rtol, atol=atol
                    ).all(), (
                        f'[ROUND {round}]  Both values of {pydtnn_layer.name_with_id}\'s "{var_key}" are not close enough: '
                        f"{torch_param=}\n{pydtnn_param=}"
                        f"({self.print_stats(torch_param, pydtnn_param, rtol, atol)})"
                    )

    @staticmethod
    def target_pydtnn2torch_format(y_pydtnn: np.ndarray) -> np.ndarray:
        """Convert PyDTNN output shape to PyTorch's format"""
        return np.argmax(y_pydtnn, axis=1)

    def do_test_model(
        self,
        model_torch: PyTorch_Model,
        model_name: str,
        model_pydtnn: PyDTNN_Model,
        without_weighted_layers: bool = False,
    ) -> None:
        """
        Executes the full comparison test for a given model.

        Args:
            model_torch (torch.nn.Module): PyTorch model to test.
            torch_forward_outputs (list[torch.Tensor]): list where the torch forward's outputs will be stored.
            model_name (str): Name of the model to test.
            model_pydtnn (pydtnn.Model): PyDTNN model to test.
        """

        params = self.params
        input_shape = params.synthetic_input_shape
        # output_shape = params.synthetic_output_shape[0]

        if params.dtype is np.dtype(np.float64):
            model_torch = model_torch.double()

        loss_func_torch = self._get_torch_loss_func()
        if not without_weighted_layers:
            optimizer_torch = self.get_optimizer_pytorch(model_torch)

        model_torch = model_torch.to(torch.device(params.device))

        x_outputs_torch = self.set_forward_hook(model_torch)
        dx_torch = self.set_backward_hook(model_torch)

        for i in range(params.num_epochs):
            if verbose_test():
                print(f"Round {i + 1}/{params.num_epochs}")

            if not without_weighted_layers:
                optimizer_torch.zero_grad()  # pyright: ignore

            x_pydtnn_base: np.ndarray = np.asarray(
                rand.random((params.batch_size, *input_shape)), dtype=params.dtype, order="C"
            )
            y_pydtnn_base: np.ndarray = np.ones(
                (params.batch_size, *model_pydtnn.output_shape), dtype=params.dtype
            )

            x_torch = torch.from_numpy(x_pydtnn_base.copy())
            if params.dtype is np.dtype(np.float64):
                x_torch = x_torch.double()
            else:
                x_torch = x_torch.float()
            x_torch.requires_grad_(True)
            x_torch = x_torch.to(torch.device(params.device))

            y_torch = (
                torch.from_numpy(self.target_pydtnn2torch_format(y_pydtnn_base).copy()).long()
            ).to(torch.device(params.device))

            input_layer = model_pydtnn.layers[0]
            assert isinstance(input_layer, Identity)
            x_pydtnn, y_pydtnn = input_layer._sync_x_y(x_pydtnn_base, y_pydtnn_base)

            # --- FORWARD ---
            if verbose_test():
                header(f"Model {model_name} 1 forward pass")
            x_torch = self.do_torch_model_forward_pass(model_torch, x_torch)

            if verbose_test():
                header(f"Model {model_pydtnn.model_name} 2 forward pass")

            model_pydtnn.real_batch_size = x_pydtnn.shape[0]
            x_pydtnn, x_to_compare = self.do_pydtnn_model_forward_pass(model_pydtnn, x_pydtnn)

            # Compare forward results
            self.compare_forward(x_outputs_torch, x_to_compare, round=i)

            # --- LOSS ---
            loss_torch = self.do_pytorch_model_loss(loss_func_torch, x_torch, y_torch)
            _loss_pydtnn, dx_pydtnn = self.do_pydtnn_model_loss(
                model_pydtnn, x_pydtnn[-1], y_pydtnn
            )
            self.compare_loss(loss_torch, _loss_pydtnn, round=i)

            # --- BACKWARD ---
            # Torch Model backward
            if verbose_test():
                header(f"Model {model_torch} 1 backward pass")

            self.do_pytorch_model_backward_pass(model_torch, loss_torch)

            # PyDTNN Model backward
            if verbose_test():
                header(f"Model {model_pydtnn.model_name} 2 backward pass")
            dx_pydtnn = self.do_pydtnn_model_backward_pass(model_pydtnn, dx_pydtnn)

            # Compare backward results
            self.compare_backward(dx_torch, dx_pydtnn, round=i)

            # Compare the "parameters.grad"/"grad_vars" results
            self.compare_grad_vars(model_torch, model_pydtnn, round=i)

            # Optimizer pass
            if not without_weighted_layers:
                self.do_pytorch_model_optimizer_pass(model_torch, optimizer_torch)  # pyright: ignore
                self.do_pydtnn_model_optimizer_pass(model_pydtnn)

                # Compare Optimizer's results
                self.compare_optimizer_parameters(model_torch, model_pydtnn, round=i)

            # Delete torch outputs:
            x_outputs_torch.clear()
            dx_torch.clear()
