from model_convertor import convert_model

from typing import Dict

from pydtnn.activations.softmax import Softmax
from pydtnn.datasets.dataset import select as select_dataset
from torchvision.models import vgg19, alexnet, densenet169, resnet50, googlenet
from torchvision.models import densenet121, densenet201, resnet18, resnet34, resnet101, resnet152, vgg11, vgg16

from torchmetrics import Accuracy, Metric

from pydtnn.datasets.dataset import Dataset

from pydtnn.models.vgg11 import vgg11 as pydtnn_vgg11
from pydtnn.models.vgg16 import vgg16 as pydtnn_vgg16
from pydtnn.models.vgg19_imagenet import vgg19_imagenet as pydtnn_vgg19_imagenet
from pydtnn.models.alexnet_cifar10 import alexnet_cifar10 as pydtnn_alexnet_cifar10
from pydtnn.models.densenet121_cifar10 import densenet121_cifar10 as pydtnn_densenet121_cifar10
from pydtnn.models.densenet169_cifar10 import densenet169_cifar10 as pydtnn_densenet169_cifar10
from pydtnn.models.densenet201_cifar10 import densenet201_cifar10 as pydtnn_densenet201_cifar10
from pydtnn.models.resnet18_cifar10 import resnet18_cifar10 as pydtnn_resnet18_cifar10
from pydtnn.models.resnet34_cifar10 import resnet34_cifar10 as pydtnn_resnet34_cifar10
from pydtnn.models.resnet50_cifar10 import resnet50_cifar10 as pydtnn_resnet50_cifar10
from pydtnn.models.resnet101_cifar10 import resnet101_cifar10 as pydtnn_resnet101_cifar10
from pydtnn.models.resnet152_cifar10 import resnet152_cifar10 as pydtnn_resnet152_cifar10
from pydtnn.models.inceptionv3_cifar10 import inceptionv3_cifar10 as pydtnn_inceptionv3_cifar10

from pydtnn.model import Model as PyDTNN_Model
from pydtnn.datasets.dataset import select as select_dataset
from pydtnn.utils.best_of import BestOf

import torch
from torch.nn import CrossEntropyLoss


dict_test = {
    "vgg11": (vgg11, pydtnn_vgg11, (524, 524, 3), "cifar10", {"num_classes": 5}, None),  # (224, 224, 3)
    "vgg16": (vgg16, pydtnn_vgg16, (524, 524, 3), "cifar10", {"num_classes": 5}, None),  # (224, 224, 3)
    "vgg19": (vgg19, pydtnn_vgg19_imagenet, (524, 524, 3), "cifar10", {"num_classes": 5}, not None),
    "alexnet": (alexnet, pydtnn_alexnet_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, None),
    "densenet121": (densenet121, pydtnn_densenet121_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, not None),
    "densenet169": (densenet169, pydtnn_densenet169_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, not None),
    "densenet201": (densenet201, pydtnn_densenet201_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, not None),
    "resnet18": (resnet18, pydtnn_resnet18_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, None),
    "resnet34": (resnet34, pydtnn_resnet34_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, None),
    "resnet50": (resnet50, pydtnn_resnet50_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, not None),
    "resnet101": (resnet101, pydtnn_resnet101_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, None),
    "resnet152": (resnet152, pydtnn_resnet152_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, not None),
    "googlenet": (googlenet, pydtnn_inceptionv3_cifar10, (524, 524, 3), "cifar10", {"num_classes": 5}, None),  # (299, 299, 3)
}

# ----- EXECUTION PARAMETERS ----- #
TEST = "densenet169"
FIRST_PYTORCH = False
OLD_FIRST = None
DATASET_PATH = ""
WEIGHTS_PATH = ""
OUTPUT_PATH = ""
INFERENCE = True
TRAINING = False

KWARGS = {
    "model_name": None,
    "dataset": None,
    "dataset_name": None,
    "evaluate_only": True,
    "parallel": "data",
    "tensor_format": "NCHW",  # "NCHW" # "NHWC",
    "loss_func": "categorical_cross_entropy",
    "enable_gpu": False,  # True,
    "dataset_train_path": DATASET_PATH,
    "dataset_test_path": DATASET_PATH,
}


def get_model_layers(model: torch.nn.Module, name: str = "self") -> Dict[str, torch.nn.Module]:
    # Recursive function to get the models without containers modules.
    def _get_model_layers(model: torch.nn.Module, name: str, dict_modules: Dict[str, torch.nn.Module]):
        # The recursive function.
        children = list(model.named_children())
        if len(children) > 0:
            for nom, module in children:
                _get_model_layers(model=module, name=".".join([name, nom]), dict_modules=dict_modules)
        else:
            dict_modules[name] = model
    # -- END _get_model_layers--#
    dict_modules = {}
    _get_model_layers(model=model, name=name, dict_modules=dict_modules)
    return dict_modules


def pytorch_inference(model: torch.nn.Module, dataloader, loss_func: torch.nn.modules.loss._Loss, device: torch.device, metrics_list: list) -> None:

    outputs_list = list()
    labels_list = list()

    print(f"metrics_list: {metrics_list}")

    if False:
        dict_layers = get_model_layers(model)
        for n in dict_layers.keys():
            m = dict_layers[n]

            def print_pre(module, x, *, name=n):
                print(f"Layer - {name}:")
                if isinstance(x[0], list):
                    for elem in x[0]:
                        print(f"\telem.size(): {elem.size()}")
                else:
                    print(f"\tx[0].size(): {x[0].size()}")
            # ----

            def print_post(module, args, output, *, name=n):
                print(f"{name}:")
                # print(f"output: {output}")
                if isinstance(output[0], list):
                    for elem in output[0]:
                        print(f"\toutput.size(): {elem.size()}")
                        # print(f"\toutput: {elem}")
                else:
                    print(f"\toutput[0].size(): {output[0].size()}")
                    # print(f"\toutput[0]: {output[0]}")
            # ----

            m.register_forward_pre_hook(print_pre)
            m.register_forward_hook(print_post)
        # --------

    model.eval()
    with torch.no_grad():

        for inputs, labels, _ in dataloader:
            inputs = torch.Tensor(inputs).to(device)
            labels = torch.Tensor(labels).to(device)
            outputs = model(inputs)
            outputs = outputs.to(device)
            # loss = loss_fn(outputs, labels)

            # print(f"outputs:\n{outputs}")
            outputs_list.extend(outputs)
            labels_list.extend(labels)
            for _, metric in metrics_list:
                metric: Metric = metric
                metric.update(outputs, labels)

    print(f"metrics_list: {metrics_list}")

    for name, metric in metrics_list:
        print(f"type(metric): {type(metric)}")
        metric_result = metric.compute()
        print(f"{name}: {metric_result:.4f}")


def print_model_reports(model):
    # Print performance counter report
    model.perf_counter.print_report()
    # Print BestOf report
    if model.enable_best_of:
        print()
        BestOf.print_report()


def pydtnn_inference(model: PyDTNN_Model, metrics_list=None, dataset=None) -> None:
    metrics_list = [f for f in model.metrics.replace(" ", "").split(",")] if metrics_list is None else metrics_list
    model.dataset = dataset if dataset is not None else model.dataset
    model.show()
    model.evaluate_dataset()
    print_model_reports(model)


def _pydtnn_inference(new_model, old_model, dataset, old_first=None):
    print("-------------------")
    print(" PyDTNN's inference")
    print("-------------------")

    match old_first:
        case True:
            print("OLD model")
            pydtnn_inference(model=old_model, dataset=dataset)
            print("NEW model")
            pydtnn_inference(model=new_model, dataset=dataset)
        case False:
            print("NEW model")
            pydtnn_inference(model=new_model, dataset=dataset)
            print("OLD model")
            pydtnn_inference(model=old_model, dataset=dataset)
        case not_old_model:  # old_first = None
            print("NEW model")
            pydtnn_inference(model=new_model, dataset=dataset)
# -----------------------#


def _pytorch_inference(pytorch_model, dataloader, kwargs, device):
    print("-------------------")
    print("Pytorch's inference")
    print("-------------------")

    match kwargs["loss_func"]:
        case "categorical_cross_entropy":
            loss = CrossEntropyLoss()
        case _:
            loss = None
            print("Pick another loss")
            assert False

    task = "binary"
    num_classes = 10
    pytorch_inference(model=pytorch_model, dataloader=dataloader, loss_func=loss, device=device,
                      metrics_list=[("Accuracy", Accuracy(task=task, num_classes=num_classes)),
                                    # ("AUROC", AUROC(task = task, num_classes = num_classes)),
                                    # ("AveragePrecision", AveragePrecision(task = task, num_classes = num_classes)),
                                    # ("F1Score", F1Score(task = task, num_classes = num_classes))
                                    ])
# -----------------------#


def pydtnn_training(model: PyDTNN_Model, dataset: Dataset, num_samples=64 * 2):

    # history = model.train(x_train=dataset._x[DatasetEnum.TRAIN][:num_samples], x_val=dataset._x[VAL][:num_samples],
    #                      y_train=dataset._y[DatasetEnum.TRAIN][:num_samples], y_val=dataset._y[VAL][:num_samples])
    history = model.train_dataset()
    print(f"history: {history}")


def main():
    test = TEST

    pytorch_model, create_pydtnn_model, shape, dataset, args, weight = dict_test[test]
    pytorch_model: torch.Module = pytorch_model(**args)

    KWARGS["dataset"] = dataset
    KWARGS["dataset_name"] = dataset
    output_shape = args["num_classes"]
    kwargs = KWARGS
    shape = shape if KWARGS["tensor_format"] == "NHWC" else (shape[2], *shape[:2])
    print(f"{shape}")

    device = torch.device("cpu")  # if kwargs["enable_gpu"] == False else torch.device("cuda")
    if weight is not None:
        weight = f"{WEIGHTS_PATH}model_{test}.pth"
        weight = torch.load(weight, weights_only=True, map_location=torch.device(device))

        pytorch_model.load_state_dict(weight, strict=False,)

    print("====================")
    print("== PyDTNN version ==")
    print("====================")

    old_model = PyDTNN_Model(**kwargs)
    create_pydtnn_model(old_model, output_shape=output_shape)
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

    # dataset = get_dataset(old_model)
    # pydtnn_inference(model=old_model, dataset = dataset)

    print("=======================")
    print("== Converted version ==")
    print("=======================")

    new_model = convert_model(model=pytorch_model, input_shape=shape,
                              default_output_activation_layer=Softmax(), **kwargs)

    print("=====================")
    print("=== MODEL CREATED ===")
    print("=====================")

    new_model.show()
    print("-----")
    print(f"type(new_model): {type(new_model)}")

    if weight is not None:
        output_path = f"{OUTPUT_PATH}model_{test}.pth"
        print(f"{output_path=}")
        new_model.store_weights_and_bias(output_path)

    print("======================\n")

    print("=======================")
    print("== Testing Inference ==")
    print("=======================")

    dataset: Dataset = select_dataset(old_model.dataset_name)(old_model)

    dataloader = list(dataset._actual_batch_generator(Dataset.Part.TRAIN))

    print("dataset:")
    print(dataset)

    if False:  # check_biases_shape
        # Both PyDTNN and PyTorch Biases have the same shape.
        print(" NEW MODEL LAYERS BIASES")
        for layer in new_model.layers:
            print(layer)
            print(f"layer.biases: {layer.biases.shape}")
        print(" OLD MODEL LAYERS BIASES")
        for layer in old_model.layers:
            print(layer)
            print(f"layer.biases: {layer.biases.shape}")

    # if TRAINING:
    #    print("OLD MODEL")
    #    pydtnn_training(model=old_model, dataset=dataset)
    #    print("NEW MODEL")
    #    pydtnn_training(model=new_model, dataset=dataset)

    if INFERENCE:
        if FIRST_PYTORCH:
            _pytorch_inference(pytorch_model, dataloader, kwargs, device)
            _pydtnn_inference(new_model, old_model, dataset)
        else:
            _pydtnn_inference(new_model, old_model, dataset, old_first=OLD_FIRST)
            _pytorch_inference(pytorch_model, dataloader, kwargs, device)


if __name__ == "__main__":
    main()
