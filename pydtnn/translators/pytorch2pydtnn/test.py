from model_convertor import convert_model
import torch
from torchvision.models import googlenet

def main():
    model_path = "/home/usuario/Documentos/Resultados/pesos/model_and_weighs_googlenet.pth"
    #old_model = torch.load(model_path, map_location=torch.device('cpu') , weights_only = False)
    _old_model:torch.nn.Module = torch.load(model_path, map_location=torch.device('cpu') , weights_only = False)
    weights = _old_model.state_dict()
    old_model = googlenet(**{"num_classes": 5})
    old_model.load_state_dict(weights, strict=False)
    
    print(f"old_model:\n{old_model}")
    print("-----")
    print(f"type(old_model): {type(old_model)}")

    graph = torch.fx.symbolic_trace(old_model)
    print(graph.code)

    print("-----\n")

    new_model = convert_model(model = old_model)

    print(f"new_model:\n{new_model}")
    print("-----")
    print(f"type(new_model): {type(new_model)}")

if __name__ == "__main__":
    main()