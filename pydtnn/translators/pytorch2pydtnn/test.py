from model_convertor import convert_model

from torchvision.models import vgg19, alexnet, densenet169, resnet50, googlenet
from torchvision.models import densenet121, densenet201, resnet18, resnet34, resnet101, resnet152, vgg11, vgg16

from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score

from pydtnn.activations import *
from pydtnn.layers import *

from pydtnn.models.vgg11 import create_vgg11
from pydtnn.models.vgg16 import create_vgg16
from pydtnn.models.vgg19_imagenet import create_vgg19_imagenet
from pydtnn.models.alexnet_cifar10 import create_alexnet_cifar10
from pydtnn.models.densenet121_cifar10 import create_densenet121_cifar10
from pydtnn.models.densenet169_cifar10 import create_densenet169_cifar10
from pydtnn.models.densenet201_cifar10 import create_densenet201_cifar10
from pydtnn.models.resnet18_cifar10 import create_resnet18_cifar10
from pydtnn.models.resnet34_cifar10 import create_resnet34_cifar10
from pydtnn.models.resnet50_cifar10 import create_resnet50_cifar10
from pydtnn.models.resnet101_cifar10 import create_resnet101_cifar10
from pydtnn.models.resnet152_cifar10 import create_resnet152_cifar10
from pydtnn.models.inceptionv3_cifar10 import create_inceptionv3_cifar10

from pydtnn.model import Model as PyDTNN_Model
from pydtnn.datasets import get_dataset
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW, PYDTNN_TENSOR_FORMAT_NHWC
from pydtnn.utils.best_of import BestOf

import torch
from torch.nn import CrossEntropyLoss

DATASET_PATH = "/home/usuario/Documentos/CIBER_CAFE/Datasets/cifar-10-batches-bin"

dict_test = {
   "vgg11": (vgg11(), create_vgg11, (224, 224, 3), "imagenet"),
   "vgg16": (vgg16(), create_vgg16, (224, 224, 3), "imagenet"),
   "vgg19": (vgg19(), create_vgg19_imagenet, (32, 32, 3), "cifar10"),
   "alexnet": (alexnet(), create_alexnet_cifar10, (32, 32, 3), "cifar10"),
   "densenet121": (densenet121(), create_densenet121_cifar10, (32, 32, 3), "cifar10"),
   "densenet169": (densenet169(), create_densenet169_cifar10, (32, 32, 3), "cifar10"),
   "densenet201": (densenet201(), create_densenet201_cifar10, (32, 32, 3), "cifar10"),
   "resnet18": (resnet18(), create_resnet18_cifar10, (32, 32, 3), "cifar10"),
   "resnet34": (resnet34(), create_resnet34_cifar10, (32, 32, 3), "cifar10"),
   "resnet50": (resnet50(), create_resnet50_cifar10, (32, 32, 3), "cifar10"),
   "resnet101": (resnet101(), create_resnet101_cifar10, (32, 32, 3), "cifar10"),
   "resnet152": (resnet152(), create_resnet152_cifar10, (32, 32, 3), "cifar10"),
   "googlenet": (googlenet(), create_inceptionv3_cifar10, (299, 299, 3), "mnist"),
}

def pytorch_inference(model: torch.nn.Module, dataloader, loss_func:torch.nn.modules.loss._Loss, device:torch.device, metrics_list: list) -> None:

    outputs_list = list()
    labels_list = list()
    
    model.eval()
    with torch.no_grad():

        for inputs, labels in dataloader:
            outputs = model(inputs)            
            outputs = outputs.to(device)
            outputs_list.extend(outputs)
            labels_list.extend(labels)
            for _, metric in metrics_list:
                metric.update(outputs, labels)
    
    for name, metric in metrics_list:
        metric_result = metric.compute()
        print(f"{name}: {metric_result:.4f}")

    print("Output | Label")
    for output, label in zip(outputs_list, labels_list):
        print(f"{output} | {label}")
    
# --- END pytorch_inference --- #

def print_model_reports(model):
    # Print performance counter report
    model.perf_counter.print_report()
    # Print BestOf report
    if model.enable_best_of:
        print()
        BestOf.print_report()

def pydtnn_inference(model: PyDTNN_Model, metrics_list = None, dataset = None) -> None:
    metrics_list = [f for f in model.metrics.replace(" ", "").split(",")] if metrics_list is None else metrics_list
    model.evaluate_dataset(dataset, model.batch_size, model.loss_func, metrics_list)
    print_model_reports(model)

# --- END pytorch_inference --- #

def main():
    test = "alexnet"
    pytorch_model, create_pydtnn_model, shape, dataset = dict_test[test]
    kwargs = dict()

    kwargs["model_name"] = None
    kwargs["comm"] = None
    kwargs["mpi_processes"] = 1
    kwargs["dataset"] = kwargs["dataset_name"] = dataset
    kwargs["evaluate_only"] = True
    kwargs["parallel"] = False
    kwargs["tensor_format"] = "NCHW" # "NCHW" # "NHWC"
    kwargs["loss_func"] = "categorical_cross_entropy"
    kwargs["enable_gpu"] = False
    kwargs["dataset_train_path"] = DATASET_PATH
    kwargs["dataset_test_path"] = DATASET_PATH

    print("====================")
    print("== PyDTNN version ==")
    print("====================")


    old_model = PyDTNN_Model(**kwargs)
    create_pydtnn_model(old_model)
    print("PyDTNN version:")
    old_model.show()
    print("-----\n")
    
    print("=====================")
    print("== PyTorch version ==")
    print("=====================")
    print(pytorch_model)
    print("-----\n")

    print("PyTorch model's forward method:")
    graph = torch.fx.symbolic_trace(pytorch_model)
    print(graph.code)

    print("-----\n")

    #dataset = get_dataset(old_model)
    #pydtnn_inference(model=old_model, dataset = dataset)

    print("=======================")
    print("== Converted version ==")
    print("=======================")

    new_model = convert_model(model = pytorch_model, input_shape=shape, kwargs=kwargs)
    
    new_model.show()
    print("-----")
    print(f"type(new_model): {type(new_model)}")

    print("======================\n")

    print("=======================")
    print("== Testing Inference ==")
    print("=======================")

    dataset = get_dataset(old_model)

    print("dataset:")
    print(dataset)

    pydtnn_inference(model=old_model, dataset = dataset)

    print("-------------------")
    print(" PyDTNN's inference")
    print("-------------------")
    
    pydtnn_inference(model=new_model, dataset = dataset)

    print("-------------------")
    print("Pytorch's inference")
    print("-------------------")

    match kwargs["loss_func"]:
        case "categorical_cross_entropy":
            loss = CrossEntropyLoss
        case _:
            loss = None
            print("Pick another loss")
            assert False
    
    device = torch.device("cpu") if kwargs["enable_gpu"] == False else torch.device("gpu")

    pytorch_inference(model=pytorch_model, dataloader=None, loss_func=loss, device=device, 
                      metrics_list=[Accuracy, AUROC, AveragePrecision, F1Score])

if __name__ == "__main__":
    main()