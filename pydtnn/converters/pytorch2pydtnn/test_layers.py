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
from pydtnn.utils.best_of import BestOf

from copy import deepcopy
from math import prod
import numpy as np

# CONSANTS
N = 2
SHAPE = (3, 20, 20) # CHW
CONV_IN_CHANNELS = SHAPE[0] # Shape format: CHW
CONV_OUT_CHANNELS = 1 # = PyTorch's Number filters
CONV_KERNEL_SIZE = (2,2)
LINEAR_IN_FEATURES = SHAPE[2]
LINEAR_OUT_FEATURES = prod((SHAPE[0], SHAPE[1]))
BATCH_NORM_IN_FEATURES = SHAPE[0]
POOL_SIZE = (2,2)
ADAPTIVE_AVG_POOL_OUT_SHAPE = (6, 6)

PYTORCH_LAYER_WEIGHTS = "weight"
PYTORCH_LAYER_BIASES = "bias"

# setting random seed
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
# ---

THRESHOLD = 1e-4
DTYPE = np.float32

DICT_SUPPORTED_LAYERS:Dict[str, nn.Module] = {
    # Activations:
    "LogSigmoid": nn.LogSigmoid(), # PyTorch is more precise ==> it can differ in elements below "e-08"
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Softmax": nn.Softmax(),
    "Tanh": nn.Tanh(), 
    # Convolutional layers:
    "Conv2d": nn.Conv2d(CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE), # PyTorch is more precise ==> it can differ in elements below "e-03" 
    # Dropout layers:
    "Dropout": nn.Dropout(), # It varies due the chosen distribution. In p=0, p=1 and testing mode they have the same results.
    # Linear layers:
    "Linear": nn.Linear(LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES), 
    # Normalization layers:
    "BatchNorm2d": nn.BatchNorm2d(BATCH_NORM_IN_FEATURES),
    "Flatten": nn.Flatten(),
    # Pooling layers:
    "MaxPool2d": nn.MaxPool2d(POOL_SIZE),
    "AvgPool2d": nn.AvgPool2d(POOL_SIZE),
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d(ADAPTIVE_AVG_POOL_OUT_SHAPE),
    # NOTE: It's also necessary to test: the addition and concatenation of layers (in torch are an operation and a function respectively)
}
# END CONSTANTS

def print_model_reports(model):
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
    
    def forward(self, x):
        return self.layer(x)
# --- END TEST_PyTorch_Model --- #

class Addition_Test_PyTorch_Model(PyTorch_Model):

    def __init__(self):
        super().__init__()
        self.op0:nn.Module = DICT_SUPPORTED_LAYERS["AdaptiveAvgPool2d"]
        self.op1:nn.Module = DICT_SUPPORTED_LAYERS["MaxPool2d"]
        self.op2:nn.Module = DICT_SUPPORTED_LAYERS["AvgPool2d"]
        self.act:nn.Module = DICT_SUPPORTED_LAYERS["Tanh"]
    
    def forward(self, x):
        dict_forwards = dict()
        ro = self.op0(x) # TODO: test removing this.
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
# --- END Addition_Test_PyTorch_Model --- #

class Concat_Test_PyTorch_Model(PyTorch_Model):

    def __init__(self):
        super().__init__()
        self.op0:nn.Module = DICT_SUPPORTED_LAYERS["AdaptiveAvgPool2d"]
        self.op1:nn.Module = DICT_SUPPORTED_LAYERS["MaxPool2d"]
        self.op2:nn.Module = DICT_SUPPORTED_LAYERS["AvgPool2d"]
        self.activation1:nn.Module = DICT_SUPPORTED_LAYERS["Sigmoid"]
        self.activation2:nn.Module = DICT_SUPPORTED_LAYERS["Softmax"]
        self.act:nn.Module = DICT_SUPPORTED_LAYERS["Tanh"]
    
    def forward(self, x):
        dict_forwards = dict()
        ro = self.op0(x) # TODO: test removing this.
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
# --- END Addition_Test_PyTorch_Model --- #

def are_all_zeros(diff: np.ndarray) -> bool:
    return not diff.any()
# --- END are_all_zeros --- #

def are_all_below_threshold(diff: np.ndarray, threshold:float = THRESHOLD) -> bool:
    return np.all(diff < threshold)
# --- END are_all_zeros --- #

def inference_pydtnn_model(model: PyDTNN_Model, dataset: np.ndarray) -> np.ndarray:
    y:np.ndarray = dataset
    for layer in model.layers:
        layer:LayerAndActivationBase
        y = layer.forward(y)

    return (y)

def test_layers(name:str, pytorch_model: TEST_PyTorch_Model, kwargs: Dict[str, Any], input_shape: Tuple[int, int, int], device: torch.device, dataset:np.ndarray) -> None:

    print(pytorch_model)
    if False: # if necessary to check PyTorch's layers parameters
        for _name, layer in pytorch_model.named_children():
            params = vars(layer)        
            print(f"layer {_name}")
            for k in params.keys():
                print(f"\t{k}: {params[k]}")    
    
    print("=======================\n== Converted version ==\n=======================")

    #print("PyTorch model's forward method:")
    #graph = torch.fx.symbolic_trace(pytorch_model)
    #print(graph.code)

    print("-----\n")

    new_model:PyDTNN_Model = convert_model(model = pytorch_model, input_shape=input_shape, kwargs=kwargs,
                              default_output_activation_layer=None)
    
    new_model.mode = TRAIN_MODE
    new_model.show()
    print("-----")

    print("======================\n")

    # Must be at two layers: Input and the testing layer.
    pydtnn_layer:LayerAndActivationBase = new_model.layers.pop()
    print("=============================\n== Checking Dataset Values ==\n=============================")

    torch_dataset = torch.from_numpy(dataset).to(device)
    
    #print(f"PyTorch dataset.shape: {torch_dataset.shape}")
    #print(f"PyDTNN dataset.shape: {dataset.shape}")
    diff = torch_dataset.numpy() - dataset
    print(f"Are equal: {are_all_zeros(diff)}")

    pytorch_state_dict = pytorch_model.layer.state_dict()
    
    pytorch_weights: None | torch.Tensor = pytorch_state_dict[PYTORCH_LAYER_WEIGHTS] if PYTORCH_LAYER_WEIGHTS in pytorch_state_dict else None
    pydtnn_weights: None | np.ndarray = pydtnn_layer.weights

    there_are_pytorch_weigths = pytorch_weights is not None
    there_are_pydtnn_weights = pydtnn_weights is not None
    #print(f"there_are_pytorch_weigths: {there_are_pytorch_weigths}")
    #print(f"there_are_pydtnn_weights: {there_are_pydtnn_weights}")
    #
    #if there_are_pytorch_weigths:
    #    print(f"pytorch_weights.shape: {pytorch_weights.shape}")
    #if there_are_pydtnn_weights:
    #    print(f"pydtnn_weights.shape: {pydtnn_weights.shape}")
    #
    if name == "Linear":
    #    print(f"pydtnn_weights.shape: {pydtnn_weights.shape}")
        pydtnn_weights = pydtnn_weights.T
    #    print(f"pydtnn_weights.T.shape: {pydtnn_weights.shape}")

    if there_are_pytorch_weigths and there_are_pydtnn_weights:
        print(f"weigths are all zeros: {are_all_zeros(pytorch_weights.numpy() - pydtnn_weights)}")

    pytorch_biases : None | torch.Tensor = pytorch_state_dict[PYTORCH_LAYER_BIASES] if PYTORCH_LAYER_BIASES in pytorch_state_dict else None
    pydtnn_biases : None | np.ndarray = pydtnn_layer.biases

    there_are_pytorch_biases = pytorch_biases is not None
    there_are_pydtnn_biases = pydtnn_biases is not None
    #print(f"there_are_pytorch_biases: {there_are_pytorch_biases}")
    #print(f"there_are_pydtnn_biases: {there_are_pydtnn_biases}")
    #
    #if there_are_pytorch_biases:
    #    print(f"pytorch_biases.shape: {pytorch_biases.shape}")
    #    print(f"pytorch_biases: {pytorch_biases}")
    #if there_are_pydtnn_biases:
    #    print(f"pydtnn_biases.shape: {pydtnn_biases.shape}")
    #    print(f"pydtnn_biases: {pydtnn_biases}")
    
    if there_are_pytorch_biases and there_are_pydtnn_biases:
        print(f"biases are all zeros: {are_all_zeros(pytorch_biases.numpy() - pydtnn_biases)}")

    print("=====================\n== Testing Forward ==\n=====================")
    print(f"pydtnn_layer: {pydtnn_layer}")
    print(f"pytorch_model: {pytorch_model}")

    pytorch_output:torch.Tensor = pytorch_model(torch_dataset)
    pydtnn_output:np.ndarray = pydtnn_layer.forward(dataset)

    pytorch_output = pytorch_output.detach().to(device).numpy()

    diff = abs(pytorch_output) - abs(pydtnn_output)
    
    print(f"Are equal: {are_all_zeros(diff)} || {name}")
    print(f"Are below the threshold ({THRESHOLD}): {are_all_below_threshold(diff)} || {name}")
    if not are_all_below_threshold(diff):
        #print(f"pytorch_output.shape: {pytorch_output.shape}")
        #print(f"pydtnn_output.shape: {pydtnn_output.shape}")
        print(f"pytorch_output:\n{pytorch_output}")
        print(f"pydtnn_output:\n{pydtnn_output}")
        print(f"pytorch_output - pydtnn_output:\n{diff}")

    print("=========================================\n")
# --- END test_layers --- #

def test_add_and_concat(name:str, pytorch_model: TEST_PyTorch_Model, kwargs: Dict[str, Any], input_shape: Tuple[int, int, int], device: torch.device, dataset:np.ndarray) -> None:

    print(pytorch_model)
    
    print("=======================\n== Converted version ==\n=======================")

    print("PyTorch model's forward method:")
    graph = torch.fx.symbolic_trace(pytorch_model)
    print(graph.code)

    print("-----\n")

    pydtnn_model:PyDTNN_Model = convert_model(model = pytorch_model, input_shape=input_shape, kwargs=kwargs,
                              default_output_activation_layer=None)
    
    #pydtnn_model.mode = EVALUATE_MODE
    pydtnn_model.show()
    torch_dataset = torch.from_numpy(dataset).to(device)
    print("-----")
    
    print("======================\n")

    print("=====================\n== Testing Forward ==\n=====================")
    print(f"pydtnn_layer: {pydtnn_model}")
    print(f"pytorch_model: {pytorch_model}")

    pytorch_output, pytorch_dict_outputs = pytorch_model(torch_dataset)
    pytorch_output:torch.Tensor
    pytorch_dict_outputs:dict[str, torch.Tensor]
    pydtnn_model.dataset = dataset
    pydtnn_output = inference_pydtnn_model(pydtnn_model, dataset)
    pydtnn_output:np.ndarray
    

    pytorch_output = pytorch_output.detach().to(device).numpy()
    #print(f"pytorch_output.shape: {pytorch_output.shape}")
    #print(f"pydtnn_output.shape: {pydtnn_output.shape}")
    #print(f"pytorch_output:\n{pytorch_output}")
    #print(f"pydtnn_output:\n{pydtnn_output}")

    diff = abs(pytorch_output) - abs(pydtnn_output)
    #print(f"pytorch_output - pydtnn_output:\n{diff}")
    print(f"Are equal: {are_all_zeros(diff)} || {name}")
    print(f"Are below the threshold ({THRESHOLD}): {are_all_below_threshold(diff)} || {name}")
    print("=========================================\n")
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
    print("\n\n\n========================\n TESTING ADD AND CONCAT \n========================")
    
    for name, model in [("Addition", Addition_Test_PyTorch_Model()),
                        ("Concat", Concat_Test_PyTorch_Model()),
                        ]:
        print(f"Testing: {name}")
        test_add_and_concat(name = name, pytorch_model = model, kwargs = kwargs, input_shape = SHAPE, device=device, dataset = deepcopy(dataset))
# --- END main --- #

if __name__ == "__main__":
    main()
