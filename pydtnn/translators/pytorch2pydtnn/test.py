from model_convertor import convert_model
import torch
from torchvision.models import googlenet, vgg19

from pydtnn.activations import *
from pydtnn.layers import *

from pydtnn.models.inceptionv3_cifar10 import create_inceptionv3_cifar10
from pydtnn.model import Model as PyDTNN_Model

def main():
    model_path = "/home/usuario/Documentos/Resultados/pesos/model_and_weighs_googlenet.pth"
    #old_model = torch.load(model_path, map_location=torch.device('cpu') , weights_only = False)
    _old_model:torch.nn.Module = torch.load(model_path, map_location=torch.device('cpu') , weights_only = False)
    weights = _old_model.state_dict()
    old_model = vgg19(**{"num_classes": 5})
    old_model.load_state_dict(weights, strict=False)

    #kwargs = dict()
    #if "model_name" not in kwargs:
    #    kwargs["model_name"] = None
    #converted_model = PyDTNN_Model(**kwargs)
    #create_inceptionv3_cifar10(converted_model)
    #print("TEST")
    #converted_model.show()
    #1/0

    print(f"old_model:\n{old_model}")
    print("-----")
    print(f"type(old_model): {type(old_model)}")

    graph = torch.fx.symbolic_trace(old_model)
    print(graph.code)

    print("-----\n")

    new_model = convert_model(model = old_model, input_shape=(524, 524, 3))

    print(f"new_model:\n{new_model}")
    print("-----")
    print(f"type(new_model): {type(new_model)}")

if __name__ == "__main__":
    main()