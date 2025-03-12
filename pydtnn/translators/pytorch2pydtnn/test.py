from model_convertor import convert_model
import torch
from torchvision.models import vgg19, alexnet, densenet169, resnet50, googlenet
from torchvision.models import VGG, AlexNet, DenseNet, ResNet, GoogLeNet


from pydtnn.activations import *
from pydtnn.layers import *

from pydtnn.models.vgg19_imagenet import create_vgg19_imagenet
from pydtnn.models.alexnet import create_alexnet
from pydtnn.models.densenet169_cifar10 import create_densenet169_cifar10
from pydtnn.models.resnet50_cifar10 import create_resnet50_cifar10
from pydtnn.models.inceptionv3_cifar10 import create_inceptionv3_cifar10
from pydtnn.model import Model as PyDTNN_Model
#54.63 MBytes    
def main():
    old_model = googlenet()
    
    match old_model:
        case VGG():
            shape = (224, 224, 3)
            func = create_vgg19_imagenet
        case AlexNet():
            shape = (227, 227, 3)
            func = create_alexnet
        case DenseNet():
            shape = (32, 32, 3)
            func = create_densenet169_cifar10
        case ResNet():
            shape = (32, 32, 3)
            func = create_resnet50_cifar10
        case GoogLeNet():
            shape = (299, 299, 3)
            func = create_inceptionv3_cifar10
        case _:
            shape = ()
            func = (lambda x : print(f"Pick other model.\nModel:\n{x}"))

    kwargs = dict()
    if "model_name" not in kwargs:
        kwargs["model_name"] = None
    converted_model = PyDTNN_Model(**kwargs)
    func(converted_model)
    print("PyDTNN version:")
    converted_model.show()

    print("====================\n====================")
    print("PyTorch version")
    print(f"old_model:\n{old_model}")
    print("-----")
    
    graph = torch.fx.symbolic_trace(old_model)
    print(graph.code)

    print("-----\n")



    new_model = convert_model(model = old_model, input_shape=shape)
    
    print("\nnew_model:")
    new_model.show()
    print("-----")
    print(f"type(new_model): {type(new_model)}")

if __name__ == "__main__":
    main()

# TODO: arreglar fallos
# Densenet 169 Falla aquí:
#     features_denseblock1_denselayer1_conv2 = self.features.denseblock1.denselayer1.conv2(features_denseblock1_denselayer1_relu2);  features_denseblock1_denselayer1_relu2 = None
# cat_1 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2], 1)