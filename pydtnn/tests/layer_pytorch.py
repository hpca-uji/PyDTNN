from unittest import skip

import numpy as np
import torch

from pydtnn.activations.arctanh import Arctanh
from pydtnn.activations.leaky_relu import LeakyRelu
from pydtnn.activations.log import Log
from pydtnn.activations.relu import Relu
from pydtnn.activations.relu6 import Relu6
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.activations.softmax import Softmax
from pydtnn.activations.tanh import Tanh
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.model import Model
from pydtnn.utils import random
from pydtnn.utils.tensor import TensorFormat, format_reshape, format_transpose
from pydtnn.tests.abstract.common import Params, TestCase, verbose_test
from pydtnn.utils.constants import Parameters


#from torch.testing._internal.common_utils import numpy_to_torch_dtype_dict
numpy_to_torch_dtype_dict = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.uint16     : torch.uint16,
    np.uint32     : torch.uint32,
    np.uint64     : torch.uint64,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

# setting random seed
SEED = 1234
random.seed(SEED)
# ---------
# ===============
# Constant values
# ===============

N = 64
C = 3
H = 524
W = 524

ADAPTIVE_AVG_POOL_OUTPUT_SIZE = (3, 3)

AVG_POOL_SHAPE = (3, 3)
AVG_POOL_PADDING = 0
AVG_POOL_STRIDE = 1

BATCH_NORMALIZATION_GAMMA = 1
BATCH_NORMALIZATION_BETA = 0
BATCH_NORMALIZATION_EPSILON = 1e-5
BATCH_NORMALIZATION_MOMENTUM_PYDTNN = 0.9
BATCH_NORMALIZATION_MOMENTUM_TORCH = BATCH_NORMALIZATION_MOMENTUM_PYDTNN
BATCH_NORMALIZATION_NUM_FEATURES = C

CONV2D_IN_C_TORCH = C
CONV2D_N_FILTERS = 5
CONV2D_FILTER_SHAPE = (4, 4)
CONV2D_PADDING = 0
CONV2D_STRIDE = 1
CONV2D_DILATION = 1

FC_OUPUT_SHAPE = (4, )
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
# ==============

# ==============
# PyTorch models
# ==============

class TorchArcTanH(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arc_tanh = torch.atanh

    def forward(self, x):
        x = self.arc_tanh(x)
        return x
# -------------


class TorchAdditionBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING,
                            stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
            torch.nn.BatchNorm2d(CONV2D_N_FILTERS, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING,
                            stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = x1 + x2
        return x
# -------------


class TorchConcatenationBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING,
                            stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
            torch.nn.BatchNorm2d(CONV2D_N_FILTERS, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING,
                            stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = torch.cat([x1, x2], dim=1)
        return x
# -------------
# ====================

class D:
    def __init__(self, b=N, c=C, h=H, w=W):
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width
# ---

class ParamsLayerPytorch(Params):
    def __init__(self, d = D()) -> None:
        super().__init__()
        self.batch_size = d.b
        self.conv_variant = "i2c"
        self.tensor_format = TensorFormat.NCHW.upper()
        self.shape = format_reshape((C, H, W), "CHW", self.tensor_format[1:])
        self.model_name = None
        self.evaluate_only = True
        self.parallel = "sequential"
        self.loss_func = "categorical_cross_entropy"
        self.enable_gpu = False
        self.omm = None
        self.dtype = np.dtype(np.float32)
        self.tracing = False
        self.tracer_output = ""
        torch.set_default_dtype(numpy_to_torch_dtype_dict[self.dtype.type])
        self.dtype = np.dtype(self.dtype)
    
    def asdict(self):
        return self.__dict__
# ----

class LayerPyTorchTestCase(TestCase):

    params = ParamsLayerPytorch()

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
    # -----

    # ======================
    # Initialization methods
    # ======================

    @staticmethod
    def get_test_data(no_zeros=False, normalize=True, positives_and_negatives=True, 
                      shape_with_elements = (params.batch_size, *params.shape), dtype = params.dtype) -> np.ndarray:
        num_elems = np.prod(shape_with_elements) // 4

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
        random.shuffle(x)

        if normalize:
            min_x = np.min(x)
            x = (x - min_x) / (np.max(x) - min_x)
            if positives_and_negatives:
                x -= 0.5

        return np.asarray(x, dtype=dtype, order="C", copy=True)
    # ---------

    @staticmethod
    def initialize_pydtnn_model(list_layers: list[LayerBase], params=params) -> Model:
        model = Model(**params.asdict())
        model.add(Input(params.shape))
        model.add_layers(list_layers)
        model.mode = Model.Mode.TRAIN
        model._initialize()
        return model
    # ---------

    def _copy_grad_vars(self, grad: np.ndarray, grad_var: str, torch_layer: torch.nn.Module) -> None:
        if grad is not None:
            torch_grad_var = GRAD_EQUIVALENCES[grad_var]
            torch_grad = getattr(torch_layer, torch_grad_var)
            torch_grad.copy_(torch.from_numpy(grad.copy()).to(torch.device("cpu")).float())
    # ---

    def copy_grad_vars(self, pydtnn_model: Model, torch_model: torch.nn.Module) -> None:
        layers = pydtnn_model.get_all_layers().copy()

        if isinstance(layers[0], Input):
            layers.pop(0)

        torch_layers = [module for module in torch_model.modules() if not isinstance(module, torch.nn.Sequential)]
        # print(f"{layers=} {len(layers)=} || {len(torch_layers)=} {torch_layers=}")

        with torch.no_grad():
            for i in range(len(layers)):
                layer = layers[i]
                torch_layer = torch_layers[i]
                match layer:
                    case BatchNormalization():
                        running_mean = layer.running_mean
                        running_var = layer.running_var
                        if running_mean is not None:
                            torch_layer.running_mean.copy_(torch.from_numpy(running_mean.copy()).to(torch.device("cpu")).float())
                        if running_var is not None:
                            torch_layer.running_var.copy_(torch.from_numpy(running_var.copy()).to(torch.device("cpu")).float())
                    case FC():
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            grad = grad if grad_var != Parameters.WEIGHTS else grad.T
                            self._copy_grad_vars(grad, grad_var, torch_layer)
                    case Conv2D():
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            if grad_var == "weights" and grad is not None:
                                grad = format_transpose(grad, {TensorFormat.NHWC: "ihwo", TensorFormat.NCHW: "oihw"}[pydtnn_model.tensor_format], "oihw")
                            self._copy_grad_vars(grad, grad_var, torch_layer)
                    case _:
                        for grad_var in layer.grad_vars.keys():
                            grad: np.ndarray = getattr(layer, grad_var)
                            self._copy_grad_vars(grad, grad_var, torch_layer)
    # ----

    def do_test(self, _x: np.ndarray, pydtnn_model: Model, torch_model: torch.nn.Module, name_test: str, rtol=1e-6, atol=1e-6) -> None:
        self.copy_grad_vars(pydtnn_model, torch_model)

        num_elems = (len("Testing: ") + len(name_test))
        if verbose_test():
            print(f"\n\n{'=' * num_elems}\nTesting: {name_test}\n{'=' * num_elems}")

        x = np.copy(_x)

        x = x.astype(dtype=self.params.dtype, order="C", copy=None)

        for layer in pydtnn_model.layers:
            x: np.ndarray = layer.forward(x)
        x_pydtnn = x
        x_pydtnn = format_transpose(x, self.params.tensor_format.upper(), TensorFormat.NCHW.upper())

        x = torch.from_numpy(_x.reshape((N, C, H, W), copy=False)).to(torch.device("cpu")).float()
        x_torch: torch.Tensor = torch_model(x)
        x_torch = np.asarray(x_torch.cpu().detach().numpy(), dtype=pydtnn_model.dtype, order="C", copy=None)

        if verbose_test():
            print(f"[{rtol=}, {atol=}]\n{x_pydtnn.max()=}\n{x_torch.max()=}\n{x_pydtnn.min()=}\n{x_torch.min()=}\n{x_pydtnn.std()=}\n{x_torch.std()=}\n{x_pydtnn.mean()=}\n{x_torch.mean()=}")

        diff = x_pydtnn - x_torch
        if verbose_test():
            print(f"{diff.max()=}\n{diff.min()=}\n{diff.std()=}\n{diff.mean()=}")

        # if not (diff < rtol).all():
        #    print(f"x_pydtnn:\n{x_pydtnn}")
        #    print(f"x_torch:\n{x_torch}")
        #    print(f"diff:\n{diff}")

        # self.assertTrue((diff < rtol).all()), f"Not all values are below the rtol. Max. difference: \"{diff.max()}\". Std. deviation: \"{diff.std()}\". Min. difference: {diff.min()}."
        self.assertTrue(np.allclose(x_pydtnn, x_torch, rtol=rtol, atol=atol))
    # ---------
    # ====================

    # ====================
    # Unitary Test methods
    # ====================

    def test_AdaptiveAveragePool2D(self):
        pydtnn_layers = [AdaptiveAveragePool2D(output_shape=ADAPTIVE_AVG_POOL_OUTPUT_SIZE)]
        torch_model = torch.nn.AdaptiveAvgPool2d(output_size=ADAPTIVE_AVG_POOL_OUTPUT_SIZE)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()

        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AdaptiveAveragePool2D", rtol=1e-4, atol=1e-3)
    # ---------

    def test_AveragePool2D(self):
        pydtnn_layers = [AveragePool2D(pool_shape=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE)]
        torch_model = torch.nn.AvgPool2d(kernel_size=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AveragePool2D")
    # ---------

    def test_BatchNormalization(self):
        pydtnn_layers = [BatchNormalization(gamma=BATCH_NORMALIZATION_GAMMA, beta=BATCH_NORMALIZATION_BETA, epsilon=BATCH_NORMALIZATION_EPSILON,
                                            momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)]
        torch_model = torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH, affine=False)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)

        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="BatchNormalization", rtol=1e0, atol=1e0)
    # ---------

    def test_Conv2D(self):
        pydtnn_layers = [Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)]
        torch_model = torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE,
                                      padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Conv2D")
    # ---------

    @skip(reason="Dropout makes a random mask, then it can not be compared due both PyTorch and PyDTNN create different masks.")
    def test_Dropout(self):
        pydtnn_layers = [Dropout()]
        torch_model = torch.nn.Dropout()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Dropout")
    # ---------

    def test_Flatten(self):
        pydtnn_layers = [Flatten()]
        torch_model = torch.nn.Flatten()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Flatten")
    # ---------

    def test_FC(self):
        pydtnn_layers = [Flatten(), FC(shape=FC_OUPUT_SHAPE)]
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=np.prod(self.params.shape), out_features=LINEAR_OUTPUT))
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="FC", rtol=1e-5, atol=1e-5)
    # ---------

    def test_MaxPool2D(self):
        pydtnn_layers = [MaxPool2D(pool_shape=MAX_POOL_SHAPE, padding=MAX_POOL_PADDING, stride=MAX_POOL_STRIDE, dilation=MAX_POOL_DILATION)]
        torch_model = torch.nn.MaxPool2d(kernel_size=MAX_POOL_SHAPE, padding=MAX_POOL_PADDING, stride=MAX_POOL_STRIDE, dilation=MAX_POOL_DILATION)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="MaxPool2D")
    # ---------

    def test_AdditionBlock(self):
        pydtnn_layers = [
            AdditionBlock(
                [Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
                 BatchNormalization(gamma=BATCH_NORMALIZATION_GAMMA, beta=BATCH_NORMALIZATION_BETA, epsilon=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)
                 ],

                [Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
                 ]
            )]

        torch_model = TorchAdditionBlock()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AdditionBlock", rtol=2, atol=2)
    # ---------

    def test_ConcatenationBlock(self):
        pydtnn_layers = [
            ConcatenationBlock(
                [Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
                 BatchNormalization(gamma=BATCH_NORMALIZATION_GAMMA, beta=BATCH_NORMALIZATION_BETA, epsilon=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)
                 ],

                [Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE,
                        padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
                 ]
            )]
        torch_model = TorchConcatenationBlock()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="ConcatenationBlock", rtol=2, atol=2)
    # ---------

    def test_Sigmoid(self):
        pydtnn_layers = [Sigmoid()]
        torch_model = torch.nn.Sigmoid()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Sigmoid")
    # ---------

    def test_Relu(self):
        pydtnn_layers = [Relu()]
        torch_model = torch.nn.ReLU()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Relu")
    # ---------

    def test_Relu6(self):
        pydtnn_layers = [Relu6()]
        torch_model = torch.nn.ReLU6()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Relu6")
    # ---------

    def test_LeakyRelu(self):
        pydtnn_layers = [LeakyRelu()]
        torch_model = torch.nn.LeakyReLU()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="LeakyRelu")
    # ---------

    def test_Tanh(self):
        pydtnn_layers = [Tanh()]
        torch_model = torch.nn.Tanh()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Tanh")
    # ---------

    def test_Arctanh(self):
        # NOTE: Domain ArcTanH: Real numbers between "]-1, 1["
        pydtnn_layers = [Arctanh()]
        torch_model = TorchArcTanH()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Arctanh")
    # ---------

    def test_Log(self):
        pydtnn_layers = [Log()]
        torch_model = torch.nn.LogSigmoid()
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        # _x = np.where(_x < 0, 1, _x)
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Log")
    # ---------

    def test_Softmax(self):
        pydtnn_layers = [Softmax()]
        torch_model = torch.nn.Softmax(dim=1)
        pydtnn_model = LayerPyTorchTestCase.initialize_pydtnn_model(pydtnn_layers, params=self.params)
        _x = LayerPyTorchTestCase.get_test_data()
        self.do_test(_x=_x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Softmax")
    # ---------
    # ====================
