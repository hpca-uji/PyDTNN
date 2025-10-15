from pydtnn.activations import *
from pydtnn.layers import *
from pydtnn.optimizers import *
from pydtnn.layers.conv_2d import GroupingEnum
import torch
import unittest

from pydtnn import Model

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    import pycuda.gpuarray as gpuarray
from pydtnn.backends.gpu import TensorGPU

# setting random seed
SEED = 1234
np.random.seed(SEED)
# ---------


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
    "dtype": np.float32,
    "tracing": False,
    "tracer_output": "",
    "batch_size": N
}

ADAPTIVE_AVG_POOL_OUTPUT_SIZE = (3, 3)

AVG_POOL_SHAPE = (3, 3)
AVG_POOL_PADDING = 0
AVG_POOL_STRIDE = 1

BATCH_NORMALIZATION_EPSILON=1e-05
BATCH_NORMALIZATION_MOMENTUM_PYDTNN= 0.9
BATCH_NORMALIZATION_MOMENTUM_TORCH= 1 - BATCH_NORMALIZATION_MOMENTUM_PYDTNN
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
        self.block1 = torch.nn.Sequential([
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, 
               stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
            torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
        ])
        self.block2 = torch.nn.Sequential([
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, 
               stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        ])

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = x1 + x2
        return x
# -------------

class TorchConcatenationBlock(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block1 = torch.nn.Sequential([
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, 
               stride=CONV2D_STRIDE, dilation=CONV2D_DILATION),
            torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
        ])
        self.block2 = torch.nn.Sequential([
            torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, 
               stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        ])

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x = torch.cat([x1, x2])
        return x
# -------------
# ====================

shape = (N, *SHAPE)
num_elems = np.prod(shape) // 4
x_1 = np.arange(num_elems)
x_2 = np.arange(num_elems) * -1
x_1_1 = np.where(x_1 % 2 == 0, x_1, x_1 / 3)
x_1_2 = np.where(x_1 % 2 != 0, x_1, x_1)
x_2_1 = np.where(x_1 % 2 == 0, x_2, x_2 / 3)
x_2_2 = np.where(x_1 % 2 != 0, x_2, x_2)
x = np.stack([x_1_1, x_1_2, x_2_1, x_2_2], axis=0, dtype=np.float32, casting="unsafe").reshape(shape)

class CheckLayerWithPyTorch(unittest.TestCase):

    # ======================
    # Initialization methods
    # ======================

    @staticmethod
    def get_test_data() -> np.ndarray:
        shape = (N, *SHAPE)
        num_elems = np.prod(shape) // 4

        x_1 = np.arange(num_elems)
        x_2 = np.arange(num_elems) * -1

        x_1_1 = np.where(x_1 % 2 == 0, x_1, x_1 / 3)
        x_1_2 = np.where(x_1 % 2 != 0, x_1, x_1)

        x_2_1 = np.where(x_1 % 2 == 0, x_2, x_2 / 3)
        x_2_2 = np.where(x_1 % 2 != 0, x_2, x_2)

        x = np.stack([x_1_1, x_1_2, x_2_1, x_2_2], axis=0, dtype=np.float32, casting="unsafe").reshape(shape, order="C", copy=None)
        np.random.shuffle(x)

        return x
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


    def do_test(self, _x: np.ndarray, pydtnn_model: Model, torch_model: torch.nn.Module) -> None:
        x = np.copy(_x)
        shape = (N, *SHAPE)

        for layer in pydtnn_model.layers:
            x_pydtnn:np.ndarray = layer.forward(x)

        x = torch.from_numpy(_x.reshape((N, C, H, W), copy=False)).to(torch.device("cpu"))
        # bn = torch.torch.nn.Conv2d(in_channels=C, out_channels=3,kernel_size=(2, 2), stride=1)

        x_torch: torch.Tensor = torch_model(x)

        x_torch = x_torch.cpu().detach().numpy().reshape(shape, order="C", copy=None)

        print(f"{x_pydtnn.shape=}")
        print(f"{x_pydtnn.shape=}")
        print(f"{x_torch.shape=}")

        print(f"x_pydtnn.max:\t{x_pydtnn.max()}")
        print(f"x_torch.max: \t{x_torch.max()}")
        print(f"x_pydtnn.min:\t{x_pydtnn.min()}")
        print(f"x_torch.min: \t{x_torch.min()}")

        threshold = 1e-6
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
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_AveragePool2D(self):
        pydtnn_layers = [AveragePool2D(pool_shape=AVG_POOL_SHAPE, padding=AVG_POOL_PADDING, stride=AVG_POOL_STRIDE)]
        torch_model = torch.nn.AvgPool2d(kernel_size=AVG_POOL_SHAPE, padding=AVG_POOL_STRIDE, stride=AVG_POOL_STRIDE)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_BatchNormalization(self):
        pydtnn_layers = [BatchNormalization(epsilon=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)]
        torch_model = torch.nn.BatchNorm2d(BATCH_NORMALIZATION_NUM_FEATURES, eps=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_TORCH)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Conv2D(self):
        pydtnn_layers = [Conv2D(grouping=GroupingEnum.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)]
        torch_model = torch.nn.Conv2d(in_channels=CONV2D_IN_C_TORCH, out_channels=CONV2D_N_FILTERS, kernel_size=CONV2D_FILTER_SHAPE, padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Dropout(self):
        pydtnn_layers = [Dropout()]
        torch_model = torch.nn.Dropout()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Flatten(self):
        pydtnn_layers = [Flatten()]
        torch_model = torch.nn.Flatten()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_FC(self):
        pydtnn_layers = [Flatten(), FC(shape=FC_OUPUT_SHAPE)]
        torch_model = torch.nn.Sequential([torch.nn.Flatten(), torch.nn.LazyLinear(out_features=LINEAR_OUTPUT)])
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_MaxPool2D(self):
        pydtnn_layers = [MaxPool2D(pool_shape=MAX_POOL_SHAPE, padding=MAX_POOL_PADDING, stride=MAX_POOL_STRIDE, dilation=MAX_POOL_DILATION)]
        torch_model = torch.nn.MaxPool2d(kernel_size=MAX_POOL_SHAPE, padding=MAX_POOL_PADDING, stride=MAX_POOL_STRIDE, dilation=MAX_POOL_DILATION)
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_AdditionBlock(self):
        pydtnn_layers = [AdditionBlock(
            [Conv2D(grouping=GroupingEnum.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, 
                    padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION), 
             BatchNormalization(epsilon=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)],

            [Conv2D(grouping=GroupingEnum.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, 
                    padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)])]
        torch_model = TorchAdditionBlock()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_ConcatenationBlock(self):
        pydtnn_layers = [ConcatenationBlock(
            [Conv2D(grouping=GroupingEnum.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, 
                    padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION), 
             BatchNormalization(epsilon=BATCH_NORMALIZATION_EPSILON, momentum=BATCH_NORMALIZATION_MOMENTUM_PYDTNN)],

            [Conv2D(grouping=GroupingEnum.STANDARD, nfilters=CONV2D_N_FILTERS, filter_shape=CONV2D_FILTER_SHAPE, 
                    padding=CONV2D_PADDING, stride=CONV2D_STRIDE, dilation=CONV2D_DILATION)])]
        torch_model = TorchConcatenationBlock()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Sigmoid(self):
        pydtnn_layers = [Sigmoid()]
        torch_model = torch.nn.Sigmoid()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Relu(self):
        pydtnn_layers = [Relu()]
        torch_model = torch.nn.ReLU()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Relu6(self):
        pydtnn_layers = [Relu6()]
        torch_model = torch.nn.ReLU6()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_LeakyRelu(self):
        pydtnn_layers = [LeakyRelu()]
        torch_model = torch.nn.LeakyReLU()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Tanh(self):
        pydtnn_layers = [Tanh()]
        torch_model = torch.nn.Tanh()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Arctanh(self):
        pydtnn_layers = [Arctanh()]
        torch_model =  TorchArcTanH()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Log(self):
        pydtnn_layers = [Log()]
        torch_model = torch.nn.LogSigmoid()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------


    def test_Softmax(self):
        pydtnn_layers = [Softmax()],
        torch_model = torch.nn.Softmax()
        pydtnn_model = CheckLayerWithPyTorch.initialize_pydtnn_model(pydtnn_layers, kwargs=KWARGS)
        _x = CheckLayerWithPyTorch.get_test_data()
        self.do_test(_x = _x, pydtnn_model=pydtnn_model, torch_model=torch_model)
    # ---------
    # ====================
