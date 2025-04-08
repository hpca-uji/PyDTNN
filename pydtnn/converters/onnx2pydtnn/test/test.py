import onnx
from pydtnn.converters.onnx2pydtnn.model_converter import convert_model

# Models from: https://github.com/onnx/models

#pth = "/home/usuario/Documentos/ONXX_pruebas/resnet50-v1-7/resnet50-v1-7.onnx"
#pth = "/home/usuario/Documentos/ONXX_pruebas/densenet169/model/model.onnx"
pth = "/home/usuario/Documentos/ONXX_pruebas/vgg19/vgg19-bn-7/vgg19-bn-7.onnx"

#print("Starting")
onnx_model = onnx.load(pth)

weights_dict = {node.name: onnx.numpy_helper.to_array(node) for node in onnx_model.graph.initializer}
for elem in weights_dict.keys():
    print(f"weights_dict[{elem}]: {weights_dict[elem]}")

weights_dict = {node.name: node.dims for node in onnx_model.graph.initializer}
inputs_dict = {_input.name: [elem.dim_value for elem in _input.type.tensor_type.shape.dim if elem.dim_value != 0] 
                    for _input in  onnx_model.graph.input if _input.name not in weights_dict.keys()}
outputs_dict = {ouput.name: [elem.dim_value for elem in ouput.type.tensor_type.shape.dim if elem.dim_value != 0]  
                for ouput in  onnx_model.graph.output}
#print(f"inputs: {inputs_dict}")
#print(f"outputs: {outputs_dict}")
#print(list(weights_dict.keys()))

for node in onnx_model.graph.node: 
    
    #print(node.op_type)
    #print(f" - input: {node.input}")
    for elem in node.input:
        pass
        #print(f"{elem}: {weights_dict[elem] if elem in weights_dict else '-'}")
    #print(f" - output: {node.output}")
    for elem in node.output:
        pass
        #print(f"{elem}: {weights_dict[elem] if elem in weights_dict else '-'}")
    #print(f" - name: {node.name}")
    for atribute in node.attribute:
        pass
        #print(f"\tatribute.name: {atribute.name}")
        #print(f"\tatribute.type: {atribute.type}")
        #print(f"\tget_node_attr_value: {onnx.helper.get_node_attr_value(node, atribute.name)}")
        
#print("\n----\n")
converted_model = convert_model(onnx_model=onnx_model)
#print("\n----\n")

#print(converted_model)