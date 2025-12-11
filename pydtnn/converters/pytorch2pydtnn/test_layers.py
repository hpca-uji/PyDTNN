from model_convertor import convert_model

from typing import Dict, Tuple, Any, Callable

from torch.nn import Module as PyTorch_Model
import torch.nn as nn
import torch

from pydtnn.model import Model as PyDTNN_Model
from pydtnn.layer_base import LayerBase
from pydtnn.utils.best_of import BestOf

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
try:
    import pycuda.gpuarray as gpuarray
    from pydtnn.libs import libcudnn as cudnn
except BaseException:
    pass

from copy import deepcopy
from math import prod
import numpy as np
from pydtnn.converters.pytorch2pydtnn.common import TRANSPOSE_WEIGHTS_LAYERS
from pydtnn.utils import random

import pydtnn

# CONSTANTS
N = 100
SHAPE = (3, 20, 20)  # NCHW
CONV_IN_CHANNELS = SHAPE[0]  # Shape format: CHW
CONV_OUT_CHANNELS = 64  # = PyTorch's Number filters
CONV_KERNEL_SIZE = (2, 2)
LINEAR_IN_FEATURES = SHAPE[2]
LINEAR_OUT_FEATURES = prod((SHAPE[0], SHAPE[1]))
BATCH_NORM_IN_FEATURES = SHAPE[0]
POOL_SIZE = (2, 2)
ADAPTIVE_AVG_POOL_OUT_SHAPE = (6, 6)

PYTORCH_LAYER_WEIGHTS = "weight"
PYTORCH_LAYER_BIASES = "bias"

# setting random seed
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
# ---

THRESHOLD = 1e-4
DTYPE = np.float32

KWARGS = {
    "model_name": None,
    # "dataset": None,
    # "dataset_name": None,
    "evaluate_only": True,
    "parallel": "data",
    "tensor_format": "NCHW",  # "NCHW" # "NHWC",
    "loss_func": "categorical_cross_entropy",
    "enable_gpu": False,  # True,
    "omm": None,
    "dtype": DTYPE,
    "tracing": False,
    "tracer_output": "",
    "batch_size": N
}

TYPES_DATA_CUDA = {np.float64: "CUDNN_DATA_DOUBLE",
                   np.float32: "CUDNN_DATA_FLOAT",
                   np.int8: "CUDNN_DATA_INT8",
                   np.int32: "CUDNN_DATA_INT32"}

DICT_SUPPORTED_LAYERS: Dict[str, Tuple[nn.Module, float]] = {
    # Activations:
    "LogSigmoid": (nn.LogSigmoid(), 1e-5),  # PyTorch is more precise ==> it can differ in elements below "e-08"
    "ReLU": (nn.ReLU(), 1e-5),
    "ReLU6": (nn.ReLU6(), 1e-5),
    "LeakyReLU": (nn.LeakyReLU(negative_slope=6), 1e-5),
    "Sigmoid": (nn.Sigmoid(), 1e-5),
    "Softmax": (nn.Softmax(), 1e-5),
    "Tanh": (nn.Tanh(), 1e-5),
    # Convolutional layers:
    "Conv2d": (nn.Conv2d(CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE), 2e-3),  # PyTorch is more precise ==> it can differ in elements below "e-03"
    # Dropout layers:
    "Dropout": (nn.Dropout(), 1e10),  # It varies due the chosen distribution. In p=0, p=1 and testing mode they have the same results.
    # Linear layers:
    # "Linear": (nn.Linear(LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES), 2e-3),
    # Normalization layers:
    "BatchNorm2d": (nn.BatchNorm2d(BATCH_NORM_IN_FEATURES), 1e-5),
    "Flatten": (nn.Flatten(), 1e-5),
    # Pooling layers:
    "MaxPool2d": (nn.MaxPool2d(POOL_SIZE), 1e-5),
    "AvgPool2d": (nn.AvgPool2d(POOL_SIZE), 1e-5),
    "AdaptiveAvgPool2d": (nn.AdaptiveAvgPool2d(ADAPTIVE_AVG_POOL_OUT_SHAPE), 1e-5),
}
# END CONSTANTS


def print_model_reports(model: PyDTNN_Model):
    # Print performance counter report
    model.perf_counter.print_report()
    # Print BestOf report
    if model.enable_best_of:
        print()
        BestOf.print_report()


class TEST_PyTorch_Model(PyTorch_Model):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        print(f"self.layer: {self.layer}")

    def forward(self, x):
        return self.layer(x)


class Addition_Test_PyTorch_Model(PyTorch_Model):

    def __init__(self):
        super().__init__()
        self.op0: nn.Module = DICT_SUPPORTED_LAYERS["AdaptiveAvgPool2d"][0]
        self.op1: nn.Module = DICT_SUPPORTED_LAYERS["MaxPool2d"][0]
        self.op2: nn.Module = DICT_SUPPORTED_LAYERS["AvgPool2d"][0]
        self.act: nn.Module = DICT_SUPPORTED_LAYERS["Tanh"][0]

    def forward(self, x):
        dict_forwards = dict()
        ro = self.op0(x)
        dict_forwards["AvgPool2d"] = ro
        r1 = self.op1(ro)
        dict_forwards["MaxPool2d"] = r1
        r2 = self.op2(ro)
        dict_forwards["AvgPool2d"] = r2
        res = r1 + r2
        dict_forwards["ADD"] = res
        res = self.act(res)
        dict_forwards["Tanh"] = res
        return (res, dict_forwards)


class Concat_Test_PyTorch_Model(PyTorch_Model):

    def __init__(self):
        super().__init__()
        self.op0: nn.Module = DICT_SUPPORTED_LAYERS["AdaptiveAvgPool2d"][0]
        self.op1: nn.Module = DICT_SUPPORTED_LAYERS["MaxPool2d"][0]
        self.op2: nn.Module = DICT_SUPPORTED_LAYERS["AvgPool2d"][0]
        self.activation1: nn.Module = DICT_SUPPORTED_LAYERS["Sigmoid"][0]
        self.activation2: nn.Module = DICT_SUPPORTED_LAYERS["Softmax"][0]
        self.act: nn.Module = DICT_SUPPORTED_LAYERS["Tanh"][0]

    def forward(self, x):
        dict_forwards = dict()
        ro = self.op0(x)
        dict_forwards["AdaptiveAvgPool2d"] = ro
        r1 = self.op1(ro)
        dict_forwards["MaxPool2d"] = r1
        r1 = self.activation1(r1)
        dict_forwards["Sigmoid"] = r1
        r2 = self.op2(ro)
        dict_forwards["AvgPool2d"] = r2
        r2 = self.activation2(r2)
        dict_forwards["Softmax"] = r2
        res = torch.concat([r1, r2], dim=1)
        dict_forwards["CONCAT"] = res
        res = self.act(res)
        dict_forwards["Tanh"] = res
        return (res, dict_forwards)


def are_all_zeros(diff: np.ndarray) -> bool:
    return not diff.any()


def are_all_below_threshold(diff: np.ndarray, threshold: float = THRESHOLD) -> bool:
    return np.all(diff < threshold)


def forward_pydtnn_model(model: PyDTNN_Model, dataset: np.ndarray | TensorGPU) -> np.ndarray | TensorGPU:
    y: np.ndarray | TensorGPU = dataset

    for i in range(1, len(model.layers)):  # NOTE - Remember: Layer 0 is the Input layer and it's ignored
        layer: LayerBase = model.layers[i]
        y = layer.forward(y)

    if y is None:
        y: TensorGPU = layer.y
    # else: Nothing special.
    print(f"y | ({type(y)})")

    return y


def test_layers_gpu(model: PyDTNN_Model, dataset: np.ndarray) -> TensorGPU:

    print(f"test_layers_gpu - model")
    model.show()
    print(f"test_layers_gpu - model\n========")
    print(f"model.dtype: {model.dtype}")

    print(f"TYPES_DATA_CUDA[model.dtype]: {TYPES_DATA_CUDA[model.dtype]}")
    dtype = model.dtype
    model.cudnn_dtype = cudnn.cudnnDataType[TYPES_DATA_CUDA[model.dtype]]
    _dataset = TensorGPU(
        gpu_arr=gpuarray.empty(shape=dataset.shape, dtype=dtype),
        tensor_format=model.tensor_format, cudnn_dtype=model.cudnn_dtype)

    _dataset.ary.set(dataset)
    print(f"_dataset: {_dataset} | type(_dataset): {type(_dataset)} | _dataset.ary.shape: {_dataset.ary.shape}")

    model.y_batch = _dataset

    y: TensorGPU | None = forward_pydtnn_model(model, _dataset)

    return y


def test_layers(name: str, pytorch_model: TEST_PyTorch_Model, kwargs: Dict[str, Any], input_shape: Tuple[int, int, int],
                device: torch.device, dataset: np.ndarray, threshold: float, function_to_test_layers: Callable) -> None:

    print(pytorch_model)

    print("=======================\n== Converted version ==\n=======================")

    new_model: PyDTNN_Model = convert_model(model=pytorch_model, input_shape=input_shape,
                                            default_output_activation_layer=None,
                                            is_input_shape_in_format=True, **kwargs)

    new_model.mode = PyDTNN_Model.Mode.TRAIN
    # new_model.show()
    # new_model.dataset = dataset
    print("-----")
    # print("PyTorch model's forward method:")
    # graph = torch.fx.symbolic_trace(pytorch_model)
    # print(graph.code)
    # print("-----\n")

    # Must be only two layers: "Input" layer and the testing one.
    pydtnn_layer: LayerBase = new_model.layers[-1]
    print("=============================\n== Checking Dataset Values ==\n=============================")

    torch_dataset = torch.from_numpy(dataset).to(device)

    diff = torch_dataset.cpu().detach().numpy() - dataset
    print(f"Are equal: {are_all_zeros(diff)}")

    pytorch_state_dict = pytorch_model.layer.state_dict()

    pytorch_weights: None | torch.Tensor = pytorch_state_dict[PYTORCH_LAYER_WEIGHTS] if PYTORCH_LAYER_WEIGHTS in pytorch_state_dict else None
    pydtnn_weights: None | np.ndarray | TensorGPU = pydtnn_layer.weights

    if isinstance(pydtnn_weights, TensorGPU):
        pydtnn_weights: np.ndarray = pydtnn_weights.ary.get()

    there_are_pytorch_weigths = pytorch_weights is not None
    there_are_pydtnn_weights = pydtnn_weights is not None

    if there_are_pytorch_weigths and there_are_pydtnn_weights:
        pydtnn_weights = pydtnn_weights.T if name in TRANSPOSE_WEIGHTS_LAYERS else pydtnn_weights
        print(f"weigths are all zeros: {are_all_zeros(pytorch_weights.cpu().detach().numpy() - pydtnn_weights)}")

    pytorch_biases: None | torch.Tensor = pytorch_state_dict[PYTORCH_LAYER_BIASES] if PYTORCH_LAYER_BIASES in pytorch_state_dict else None
    pydtnn_biases: None | np.ndarray | TensorGPU = pydtnn_layer.biases

    if isinstance(pydtnn_weights, TensorGPU):
        pydtnn_biases: TensorGPU = pydtnn_biases.ary.get()

    there_are_pytorch_biases = pytorch_biases is not None
    there_are_pydtnn_biases = pydtnn_biases is not None

    if there_are_pytorch_biases and there_are_pydtnn_biases:
        print(f"biases are all zeros: {are_all_zeros(pytorch_biases.cpu().detach().numpy() - pydtnn_biases)}")

    print("=====================\n== Testing Forward ==\n=====================")
    print(f"pytorch_model: {pytorch_model}")

    pytorch_output: torch.Tensor = pytorch_model(torch_dataset)
    pydtnn_output: np.ndarray = function_to_test_layers(model=new_model, dataset=dataset)

    pytorch_output = pytorch_output.detach().to("cpu").numpy()

    diff = abs(pytorch_output) - abs(pydtnn_output)

    are_below_threshold = are_all_below_threshold(diff, threshold)
    print(f"Are equal: {are_all_zeros(diff)} || {name}")
    print(f"Are below the threshold ({threshold}): {are_below_threshold} || {name}")
    print(f"Min. value: {np.min(pydtnn_output)}")
    print(f"Max. value: {np.max(pydtnn_output)}")
    print(f"Mean of the values: {np.mean(pydtnn_output)}")
    print(f"Median of the values: {np.median(pydtnn_output)}")
    print(f"Max. difference between outputs: {np.max(diff)}")
    if not are_below_threshold:
        print(f"pytorch_output.shape: {pytorch_output.shape}")
        print(f"pydtnn_output.shape: {pydtnn_output.shape}")
        print(f"pytorch_output:\n{pytorch_output}\n[pytorch_output]")
        print(f"pydtnn_output:\n{pydtnn_output}\n[pydtnn_output]")
        print(f"pytorch_output - pydtnn_output:\n{diff}\n[pytorch_output - pydtnn_output]")

    print("=========================================\n")


def test_add_and_concat(name: str, pytorch_model: TEST_PyTorch_Model, kwargs: Dict[str, Any], input_shape: Tuple[int, int, int],
                        device: torch.device, dataset: np.ndarray, threshold: float = THRESHOLD) -> None:

    print(pytorch_model)

    print("=======================\n== Converted version ==\n=======================")

    # print("PyTorch model's forward method:")
    # graph = torch.fx.symbolic_trace(pytorch_model)
    # print(graph.code)

    print("-----\n")

    pydtnn_model: PyDTNN_Model = convert_model(model=pytorch_model, input_shape=input_shape,
                                               default_output_activation_layer=None,
                                               is_input_shape_in_format=True, **kwargs)

    # pydtnn_model.mode = ModelModeEnum.EVALUATE
    # pydtnn_model.show()
    torch_dataset = torch.from_numpy(dataset).to(device)
    print("-----")

    print("======================\n")

    print("=====================\n== Testing Forward ==\n=====================")
    print(f"pytorch_model: {pytorch_model}")

    pytorch_output, _ = pytorch_model(torch_dataset)
    pytorch_output: torch.Tensor
    pydtnn_model.dataset = dataset
    pydtnn_output = forward_pydtnn_model(pydtnn_model, dataset)
    pydtnn_output: np.ndarray

    pytorch_output = pytorch_output.detach().to("cpu").numpy()

    diff = abs(pytorch_output) - abs(pydtnn_output)
    # print(f"pytorch_output - pydtnn_output:\n{diff}")
    are_below_threshold = are_all_below_threshold(diff, threshold)
    print(f"Are equal: {are_all_zeros(diff)} || {name}")
    print(f"Are below the threshold ({threshold}): {are_below_threshold} || {name}")
    print(f"Min. value: {np.min(pydtnn_output)}")
    print(f"Max. value: {np.max(pydtnn_output)}")
    print(f"Median of the differences between outputs: {np.median(diff)}")
    print(f"Max. difference between outputs: {np.max(diff)}")
    if False and not are_below_threshold:
        print(f"pytorch_output:\n{pytorch_output}")
        print(f"pydtnn_output:\n{pydtnn_output}")
        print(f"pytorch_output - pydtnn_output:\n{diff}")
    print("=========================================\n")


def main():

    kwargs = KWARGS
    quarter_elements = prod((N, *SHAPE)) / 4

    device = torch.device("cpu") if kwargs["enable_gpu"] == False else torch.device("cuda")
    dataset_p = np.arange(quarter_elements, dtype=DTYPE) / 3
    dataset_p_int = np.arange(quarter_elements, dtype=DTYPE)
    dataset_n = np.arange(quarter_elements, dtype=DTYPE) * (-1 / 3)
    dataset_n_int = np.arange(quarter_elements, dtype=DTYPE) * (-1)

    dataset = np.concat([dataset_p, dataset_p_int, dataset_n, dataset_n_int]).reshape((N, *SHAPE))

    function_to_test_layers = (test_layers_gpu if device.type == "cuda" else forward_pydtnn_model)
    for name in DICT_SUPPORTED_LAYERS.keys():
        layer, threshold = DICT_SUPPORTED_LAYERS[name]
        model = TEST_PyTorch_Model(layer)
        print(f"Testing: {name}")
        print(f"{dataset.shape=}")
        print(f"{dataset.min()=}")
        print(f"{dataset.max()=}\n")

        test_layers(name=name, pytorch_model=model, kwargs=kwargs, input_shape=SHAPE,
                    device=device, dataset=np.copy(dataset), threshold=threshold,
                    function_to_test_layers=function_to_test_layers
                    )

    print("\n\n\n========================\n TESTING ADD AND CONCAT \n========================")

    for name, model in [("Addition", Addition_Test_PyTorch_Model()),
                        ("Concat", Concat_Test_PyTorch_Model()),
                        ]:
        print(f"Testing: {name}")
        test_add_and_concat(name=name, pytorch_model=model, kwargs=kwargs, input_shape=SHAPE, device=device, dataset=deepcopy(dataset))


if __name__ == "__main__":
    main()
