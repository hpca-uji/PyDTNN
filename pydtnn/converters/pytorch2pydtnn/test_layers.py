from model_convertor import convert_model

from typing import Dict, Tuple, Any

from torch.nn import Module as PyTorch_Model
import torch.nn as nn
import torch

from pydtnn.model import EVALUATE_MODE, TRAIN_MODE
from pydtnn.model import Model as PyDTNN_Model
from pydtnn.activations import *
from pydtnn.layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase

from copy import deepcopy
from math import prod
import numpy as np


# CONSANTS
N = 2
SHAPE = (3, 3, 3) # CHW
CONV_IN_CHANNELS = SHAPE[0] # Shape format: CHW
CONV_OUT_CHANNELS = 1 # = PyTorch's Number filters
CONV_KERNEL_SIZE = (2,2)
LINEAR_IN_FEATURES = SHAPE[2] # TODO: Check if this is correct.
LINEAR_OUT_FEATURES = prod((SHAPE[0], SHAPE[1])) # TODO: Check if this is correct.
BATCH_NORM_IN_FEATURES = SHAPE[0]
POOL_SIZE = (2,2)
ADPATIVE_AVG_POOL_OUT_SHAPE = (4, 4)

PYTORCH_LAYER_WEIGHTS = "weight"
PYTORCH_LAYER_BIASES = "bias"

# setting random seed
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
# ---

DTYPE = np.float32

DICT_SUPPORTED_LAYERS:Dict[str, nn.Module] = {
    ## Activations:
    "LogSigmoid": nn.LogSigmoid(), # Possibly correct <== (PyDTNN is more precise ==> it can differ in elements below "e-08")
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Softmax": nn.Softmax(),
    "Tanh": nn.Tanh(), # Check what happens here (Results very different)
    # Convolutional layers:
    "Conv2d": nn.Conv2d(CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE), # Seems fine.
    # Dropout layers:
    "Dropout": nn.Dropout(), 
    # Linear layers:
    "Linear": nn.Linear(LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES), 
    # Normalization layers:
    "BatchNorm2d": nn.BatchNorm2d(BATCH_NORM_IN_FEATURES),
    "Flatten": nn.Flatten(),
    # Pooling layers:
    "MaxPool2d": nn.MaxPool2d(POOL_SIZE),
    "AvgPool2d": nn.AvgPool2d(POOL_SIZE),
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d(ADPATIVE_AVG_POOL_OUT_SHAPE), # Check output
    # NOTE: It's also necessary to test: the addition and concatenation of layers (in torch are an operation and a function)
}
# END CONSTANTS

class TEST_PyTorch_Model(PyTorch_Model):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        return self.layer(x)
# --- END TEST_PyTorch_Model --- #

def are_all_zeros(diff: np.ndarray) -> bool:
    return not diff.any()
# --- END are_all_zeros --- #

def test_layers(name:str, pytorch_model: TEST_PyTorch_Model, kwargs: Dict[str, Any], input_shape: Tuple[int, int, int], device: torch.device, dataset:np.ndarray) -> None:

    print(pytorch_model)
    for _name, layer in pytorch_model.named_children():
        params = vars(layer)        
        print(f"layer {_name}")
        for k in params.keys():
            print(f"\t{k}: {params[k]}")    
    
    print("=======================")
    print("== Converted version ==")
    print("=======================")

    print("PyTorch model's forward method:")
    graph = torch.fx.symbolic_trace(pytorch_model)
    print(graph.code)

    print("-----\n")

    new_model:PyDTNN_Model = convert_model(model = pytorch_model, input_shape=input_shape, kwargs=kwargs,
                              default_output_activation_layer=None)
    
    new_model.mode = TRAIN_MODE
    new_model.show()
    print("-----")

    print("======================\n")

    # Must be at two layers: Input and the testing layer.
    pydtnn_layer:LayerAndActivationBase = new_model.layers.pop()
    print("=====================")
    print("== Checkig  Values ==")
    print("=====================")

    torch_dataset = torch.from_numpy(dataset).to(device)
    
    print(f"PyTorch dataset.shape: {torch_dataset.shape}")
    print(f"PyDTNN dataset.shape: {dataset.shape}")
    diff = torch_dataset.numpy() - dataset
    print(f"Are equal: {are_all_zeros(diff)} || {name}") # are_all_zeros(diff) === all diff elements are 0.

    pytorch_state_dict = pytorch_model.layer.state_dict()
    
    pytorch_weights: None | torch.Tensor = pytorch_state_dict[PYTORCH_LAYER_WEIGHTS] if PYTORCH_LAYER_WEIGHTS in pytorch_state_dict else None
    pydtnn_weights: None | np.ndarray = pydtnn_layer.weights

    there_are_pytorch_weigths = pytorch_weights is not None
    there_are_pydtnn_weights = pydtnn_weights is not None
    print(f"there_are_pytorch_weigths: {there_are_pytorch_weigths}")
    print(f"there_are_pydtnn_weights: {there_are_pydtnn_weights}")
    
    if there_are_pytorch_weigths:
        print(f"pytorch_weights.shape: {pytorch_weights.shape}")
    if there_are_pydtnn_weights:
        print(f"pydtnn_weights.shape: {pydtnn_weights.shape}")
    
    if name == "Linear":
        print(f"pydtnn_weights.shape: {pydtnn_weights.shape}")
        pydtnn_weights = pydtnn_weights.T
        print(f"pydtnn_weights.T.shape: {pydtnn_weights.shape}")

    if there_are_pytorch_weigths and there_are_pydtnn_weights:
        print(f"weigths are all zeros: {are_all_zeros(pytorch_weights.numpy() - pydtnn_weights)}")

    pytorch_biases : None | torch.Tensor = pytorch_state_dict[PYTORCH_LAYER_BIASES] if PYTORCH_LAYER_BIASES in pytorch_state_dict else None
    pydtnn_biases : None | np.ndarray = pydtnn_layer.biases

    there_are_pytorch_biases = pytorch_biases is not None
    there_are_pydtnn_biases = pydtnn_biases is not None
    print(f"there_are_pytorch_biases: {there_are_pytorch_biases}")
    print(f"there_are_pydtnn_biases: {there_are_pydtnn_biases}")
    
    if there_are_pytorch_biases:
        print(f"pytorch_biases.shape: {pytorch_biases.shape}")
        print(f"pytorch_biases: {pytorch_biases}")
    if there_are_pydtnn_biases:
        print(f"pydtnn_biases.shape: {pydtnn_biases.shape}")
        print(f"pydtnn_biases: {pydtnn_biases}")
    
    if there_are_pytorch_biases and there_are_pydtnn_biases:
        print(f"biases are all zeros: {are_all_zeros(pytorch_biases.numpy() - pydtnn_biases)}")

    print("=====================")
    print("== Testing Forward ==")
    print("=====================")
    print(f"pydtnn_layer: {pydtnn_layer}")
    print(f"pytorch_model: {pytorch_model}")

    pytorch_output:torch.Tensor = pytorch_model(torch_dataset)
    pydtnn_output:np.ndarray = pydtnn_layer.forward(dataset)

    pytorch_output = pytorch_output.detach().to(device).numpy()
    print(f"pytorch_output.shape: {pytorch_output.shape}")
    print(f"pydtnn_output.shape: {pydtnn_output.shape}")
    print(f"pytorch_output:\n{pytorch_output}")
    print(f"pydtnn_output:\n{pydtnn_output}")

    diff = abs(pytorch_output) - abs(pydtnn_output)
    print(f"pytorch_output - pydtnn_output:\n{diff}")
    print(f"Are equal: {are_all_zeros(diff)} || {name}")
    print("=========================================")
# --- END test_layers --- #

def main():

    kwargs = {
        "model_name": None,
        "comm": None,
        "mpi_processes": 1,
        "evaluate_only": True,
        "parallel": "sequential",
        "tensor_format": "NCHW", # "NCHW", "NHWC",
        "loss_func": "categorical_cross_entropy",
        "enable_gpu": False,
    }

    device = torch.device("cpu") if kwargs["enable_gpu"] == False else torch.device("gpu")
    dataset = np.arange(prod((N, *SHAPE)), dtype=DTYPE).reshape((N, *SHAPE))
    
    for name in DICT_SUPPORTED_LAYERS.keys():
        layer = DICT_SUPPORTED_LAYERS[name]
        model = TEST_PyTorch_Model(layer)
        print(f"Testing: {name}")
        test_layers(name = name, pytorch_model = model, kwargs = kwargs, input_shape = SHAPE, device=device, dataset = deepcopy(dataset))

# --- END main --- #

if __name__ == "__main__":
    main()
