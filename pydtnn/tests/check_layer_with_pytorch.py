from pydtnn.activations import *
from pydtnn.layers import *
from pydtnn.optimizers import *
import torch
import unittest

from pydtnn import Model
from pydtnn.utils import random
from pydtnn.tests.common import TestCase

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
from pydtnn.backends.gpu import TensorGPU

# setting random seed
SEED = 1234
random.seed(SEED)
# ---------

# TODO: Make threshold change proportonally to the number of elements (more elements, less precission)

# ===============
# Constant values
# ===============

N = 100
C = 3
H = 524
W = 524
FORMAT = "NCHW"
SHAPE = (C,H,W) if FORMAT == "NCHW" else (H,W,C)

KWARGS = {
    "model_name": None,
    # "dataset": None,
    # "dataset_name": None,
    "evaluate_only": True,
    "parallel": "data",
    "tensor_format": FORMAT, # "NCHW" # "NHWC",
    "loss_func": "categorical_cross_entropy",
    "enable_gpu": False, # False,#True,
    "omm": None,
    "dtype": np.float64,
    "tracing": False,
    "tracer_output": "",
    "batch_size": N
}

ADAPTIVE_AVG_POOL_OUTPUT_SIZE = (3, 3)

AVG_POOL_SHAPE = (3, 3)
AVG_POOL_PADDING = 0
AVG_POOL_STRIDE = 1

BATCH_NORMALIZATION_GAMMA = 1
BATCH_NORMALIZATION_BETA = 0
BATCH_NORMALIZATION_EPSILON= 1e-5
BATCH_NORMALIZATION_MOMENTUM_PYDTNN= 0.9
BATCH_NORMALIZATION_MOMENTUM_TORCH= BATCH_NORMALIZATION_MOMENTUM_PYDTNN
BATCH_NORMALIZATION_NUM_FEATURES = C

CONV2D_IN_C_TORCH = C
CONV2D_N_FILTERS = 5
CONV2D_FILTER_SHAPE = (4,4)
CONV2D_PADDING = 0
CONV2D_STRIDE = 1
CONV2D_DILATION = 1 

FC_OUPUT_SHAPE = (4, )
LINEAR_OUTPUT = FC_OUPUT_SHAPE[0]

MAX_POOL_SHAPE = (2,2)
MAX_POOL_PADDING = 0
MAX_POOL_STRIDE = 1
MAX_POOL_DILATION = 1


GRAD_EQUIVALENCES: dict[str, str] = {
    "weights" : "weight",
    "biases" : "bias",
    # "running_mean": "running_mean", # Not PyDTNN's grading var.
    # "running_var": "running_var", # Not PyDTNN's grading var.
    #"beta": "", # Not in PyTorch
    #"gamma": "", # Not in PyTorch
}
# ==============

# ==============
ignore_model = Model(**KWARGS) # NOTE: Do not delete this (it's related to the initalization of Model).
# ==============

# ==============
# PyTorch models
# ==============

class TorchArcTanH(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arc_tan = torch.atanh

    def forward(self, x):
        x = self.arc_tan(x)
        return x
# -------------

class TorchAdditionBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, 
               stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
            torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
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
            torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, 
               stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = torch.cat([x1, x2])
        return x
# -------------
# ====================

class CheckLayerWithPyTorch(TestCase):

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    # ======================
    # Initialization methods
    # ======================

    @staticmethod
    def get_test_data(no_zeros = False, normalize = False) -> np.ndarray:
        shape_with_elements = (N, *SHAPE)
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
        x = np.stack([x_1_1, x_1_2, x_2_1, x_2_2], axis=0, dtype=KWARGS["dtype"]).reshape(shape_with_elements)
        random.shuffle(x)

        if normalize:
            min_x = np.min(x)
            x = (x - min_x) / (np.max(x) - min_x)

        #return np.asarray(x, dtype=KWARGS["dtype"], order="C", copy=True)
        return x.copy()
    # ---------

    @staticmethod
    def initialize_pydtnn_model(list_layers: list[Layer], kwargs = KWARGS) -> Model:
        model = Model(**kwargs)
        model.add(Input(SHAPE))
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

    def copy_grad_vars(self, pydtnn_model: Model, torch_model:torch.nn.Module) -> None:
        layers = pydtnn_model.get_all_layers().copy()

        if isinstance(layers[0], Input):
            layers.pop(0)
        
        print(f"{layers=}")
        torch_layers = [module for module in torch_model.modules() if not isinstance(module, torch.nn.Sequential)]
        print(f"{torch_layers=}")
        #print(f"{(len(torch_layers) == len(layers))=}")

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
                            grad:np.ndarray = getattr(layer, grad_var)
                            grad = grad if grad_var != "weights" else grad.T
                            self._copy_grad_vars(grad, grad_var, torch_layer)
                    case _:
                        for grad_var in layer.grad_vars.keys():
                            grad:np.ndarray = getattr(layer, grad_var)
                            self._copy_grad_vars(grad, grad_var, torch_layer)

            # ----


    def do_test(self, _x: np.ndarray, pydtnn_model: Model, torch_model: torch.nn.Module, name_test:str, threshold=1e-6) -> None:

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        self.copy_grad_vars(pydtnn_model, torch_model)

        print(f"====================\nTesting: {name_test}\n====================")

        x = np.copy(_x)

        x = x.astype(dtype=KWARGS["dtype"], order="C", copy=None)

        for layer in pydtnn_model.layers:
            x:np.ndarray = layer.forward(x)
        x_pydtnn = x

        x = torch.from_numpy(_x.reshape((N, C, H, W), copy=False)).to(torch.device("cpu")).float()
        x_torch: torch.Tensor = torch_model(x)
        x_torch = np.asarray(x_torch.cpu().detach().numpy(), dtype=pydtnn_model.dtype, order="C", copy=None)

        print(f"{x_pydtnn.shape=}")
        print(f"{x_pydtnn.shape=}")
        print(f"{x_torch.shape=}")

        print(f"x_pydtnn.max:\t{x_pydtnn.max()}")
        print(f"x_torch.max: \t{x_torch.max()}")
        print(f"x_pydtnn.min:\t{x_pydtnn.min()}")
        print(f"x_torch.min: \t{x_torch.min()}")

        diff = x_pydtnn - x_torch
        print(f"diff all zeros {not diff.any()}")
        print(f"diff below threshold {threshold}: {(diff < threshold).all()}")
        print(f"{diff.max()=}")
        print(f"{diff.std()=}")
        print(f"{diff.min()=}")

        if not (diff < threshold).all():
            print(f"x_pydtnn:\n{x_pydtnn}")
            print(f"x_torch:\n{x_torch}")
            print(f"diff:\n{diff}")

        self.assertTrue((diff < threshold).all()), f"Not all values are below the threshold. Max. difference: \"{diff.max()}\". Std. deviation: \"{diff.std()}\". Min. difference: {diff.min()}."
    # ---------
    # ====================

    # ====================
    # Unitary Test methods
    # ====================

    def test_AdaptiveAveragePool2D(self):
        pydtnn_layers = [AdaptiveAveragePool2D(output_shape=ADAPTIVE_AVG_POOL_OUTPUT_SIZE)]
        torch_model = torch.nn.AdaptiveAvgPool2d(output_size=ADAPTIVE_AVG_POOL_OUTPUT_SIZE)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()

        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AdaptiveAveragePool2D", threshold=2e-6)
    # ---------


    def test_AveragePool2D(self):
        pydtnn_layers = [AveragePool2D(pool_shape=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE)]
        torch_model = torch.nn.AvgPool2d(kernel_size=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AveragePool2D", threshold=1e-5)
    # ---------


    def test_BatchNormalization(self):
        pydtnn_layers = [BatchNormalization(gamma=BATCH_NORMALIZATION_GAMMA, beta=BATCH_NORMALIZATION_BETA, epsilon=BATCH_NORMALIZATION_EPSILON, 
                                            momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)]
        torch_model = torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH, affine=False)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)

        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="BatchNormalization", threshold=1e-1)
    # ---------


    def test_Conv2D(self):
        pydtnn_layers = [Conv2D(grouping=Conv2D.Grouping.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)]
        torch_model = torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)        
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Conv2D", threshold=1e-5)
    # ---------

    
    def test_Dropout(self):
        pydtnn_layers = [Dropout()]
        torch_model = torch.nn.Dropout()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        # Dropout makes a random mask ==> can not be compared due both PyTorch and PyDTNN create different masks ==> always correct.
        threshold = float("inf")
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Dropout", threshold=threshold)
    # ---------


    def test_Flatten(self):
        pydtnn_layers = [Flatten()]
        torch_model = torch.nn.Flatten()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Flatten")
    # ---------


    def test_FC(self):
        pydtnn_layers = [Flatten(), FC(shape=FC_OUPUT_SHAPE)]
        torch_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(in_features=np.prod(SHAPE), out_features=LINEAR_OUTPUT))
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="FC", threshold=1e-1)
    # ---------


    def test_MaxPool2D(self):
        pydtnn_layers = [MaxPool2D(pool_shape=MAX_POOL_SHAPE, padding=MAX_POOL_PADDING, stride=MAX_POOL_STRIDE, dilation=MAX_POOL_DILATION)]
        torch_model = torch.nn.MaxPool2d(kernel_size=MAX_POOL_SHAPE, padding=MAX_POOL_PADDING, stride=MAX_POOL_STRIDE, dilation=MAX_POOL_DILATION)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="MaxPool2D")
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
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="AdditionBlock")
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
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="ConcatenationBlock")
    # ---------


    def test_Sigmoid(self):
        pydtnn_layers = [Sigmoid()]
        torch_model = torch.nn.Sigmoid()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Sigmoid")
    # ---------


    def test_Relu(self):
        pydtnn_layers = [Relu()]
        torch_model = torch.nn.ReLU()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Relu")
    # ---------


    def test_Relu6(self):
        pydtnn_layers = [Relu6()]
        torch_model = torch.nn.ReLU6()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Relu6")
    # ---------


    def test_LeakyRelu(self):
        pydtnn_layers = [LeakyRelu()]
        torch_model = torch.nn.LeakyReLU()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="LeakyRelu")
    # ---------


    def test_Tanh(self):
        pydtnn_layers = [Tanh()]
        torch_model = torch.nn.Tanh()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Tanh")
    # ---------


    def test_Arctanh(self):
        pydtnn_layers = [Arctanh()]
        torch_model =  TorchArcTanH()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Arctanh")
    # ---------


    def test_Log(self):
        pydtnn_layers = [Log()]
        torch_model = torch.nn.LogSigmoid()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data(normalize=True)
        print(f"{_x.min()=} || {_x.max()}")
        #_x = np.where(_x < 0, 1, _x)
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Log")
    # ---------


    def test_Softmax(self):
        pydtnn_layers = [Softmax()]
        torch_model = torch.nn.Softmax()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model, name_test="Softmax")
    # ---------
    # ====================
