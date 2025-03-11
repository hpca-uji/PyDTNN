from model_convertor import convert_model
import torch
from torchvision.models import vgg19, alexnet, densenet169, resnet50, googlenet

from pydtnn.activations import *
from pydtnn.layers import *

from pydtnn.models.vgg19_imagenet import create_vgg19_imagenet
from pydtnn.models.alexnet import create_alexnet
from pydtnn.models.densenet169_cifar10 import create_densenet169_cifar10
from pydtnn.models.resnet50_cifar10 import create_resnet50_cifar10
from pydtnn.models.inceptionv3_imagenet import create_inceptionv3_cifar10
from pydtnn.model import Model as PyDTNN_Model

def main():
    model_path = "/home/usuario/Documentos/Resultados/pesos/model_and_weighs_googlenet.pth"
    #old_model = torch.load(model_path, map_location=torch.device('cpu') , weights_only = False)
    #_old_model:torch.nn.Module = torch.load(model_path, map_location=torch.device('cpu') , weights_only = False)
    #weights = _old_model.state_dict()
    #old_model = googlenet(**{"num_classes" : 5})
    #old_model.load_state_dict(weights, strict=False)
    old_model = densenet169()

    kwargs = dict()
    if "model_name" not in kwargs:
        kwargs["model_name"] = None
    converted_model = PyDTNN_Model(**kwargs)
    create_densenet169_cifar10(converted_model)
    print("TEST")
    converted_model.show()

    #print(f"old_model:\n{old_model}")
    #print("-----")

    graph = torch.fx.symbolic_trace(old_model)
    print(graph.code)

    print("-----\n")

    new_model = convert_model(model = old_model, input_shape=(224, 224, 3))

    print(f"new_model:\n{new_model.show()}")
    print("-----")
    print(f"type(new_model): {type(new_model)}")

if __name__ == "__main__":
    main()

# TODO: arreglar fallos
# Densenet 169 Falla aquí:
#     features_denseblock1_denselayer1_conv2 = self.features.denseblock1.denselayer1.conv2(features_denseblock1_denselayer1_relu2);  features_denseblock1_denselayer1_relu2 = None
# cat_1 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2], 1)