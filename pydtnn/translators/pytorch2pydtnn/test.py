from model_convertor import convert_model
import torch
from torchvision.models import vgg19, alexnet, densenet169, resnet50, googlenet
from torchvision.models import densenet121, densenet201, resnet18, resnet34, resnet101, resnet152, vgg11, vgg16
from torchvision.models import VGG, AlexNet, DenseNet, ResNet, GoogLeNet


from pydtnn.activations import *
from pydtnn.layers import *

from pydtnn.models.vgg11 import create_vgg11
from pydtnn.models.vgg16 import create_vgg16
from pydtnn.models.vgg19_imagenet import create_vgg19_imagenet
from pydtnn.models.alexnet import create_alexnet
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

dict_test = {
   "vgg11": (vgg11(), create_vgg11, (224, 224, 3)),
   "vgg16": (vgg16(), create_vgg16, (224, 224, 3)),
   "vgg19": (vgg19(), create_vgg19_imagenet, (224, 224, 3)), 
   "alexnet": (alexnet(), create_alexnet, (227, 227, 3)), 
   "densenet121": (densenet121(), create_densenet121_cifar10, (32, 32, 3)),
   "densenet169": (densenet169(), create_densenet169_cifar10, (32, 32, 3)), 
   "densenet201": (densenet201(), create_densenet201_cifar10, (32, 32, 3)),   
   "resnet18": (resnet18(), create_resnet18_cifar10, (32, 32, 3)),
   "resnet34": (resnet34(), create_resnet34_cifar10, (32, 32, 3)),
   "resnet50": (resnet50(), create_resnet50_cifar10, (32, 32, 3)), 
   "resnet101": (resnet101(), create_resnet101_cifar10, (32, 32, 3)),
   "resnet152": (resnet152(), create_resnet152_cifar10, (32, 32, 3)),
   "googlenet": (googlenet(), create_inceptionv3_cifar10, (299, 299, 3))   
}

def main():
    test = "resnet18"
    old_model, create_pydtnn_model,shape = dict_test[test]

    kwargs = dict()
    if "model_name" not in kwargs:
        kwargs["model_name"] = None
    _old_model = PyDTNN_Model(**kwargs)
    create_pydtnn_model(_old_model)
    print("PyDTNN version:")
    _old_model.show()

    print("====================\n====================")
    print("PyTorch version")
    print(f"old_model:\n{old_model}")
    print("-----")
    

    print("-----\n")

    print("old_model's forward function:")
    graph = torch.fx.symbolic_trace(old_model)
    print(graph.code)

    print("======\n")

    new_model = convert_model(model = old_model, input_shape=shape)
    
    print("\nnew_model:")
    new_model.show()
    print("-----")
    print(f"type(new_model): {type(new_model)}")

if __name__ == "__main__":
    main()