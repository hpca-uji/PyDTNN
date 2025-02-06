# Typing related
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
import numpy as np

# Operations/transformations related
import onnx
from model import Model as PyDTNN_Model
from constants import SWITCH_OPERATION_ONNX_TO_PYDTNN, CONST_NODE, CONST_OPSET, CONST_OUPTUS, CONST_ATTRIBUTES
from pydtnn.layers import Input

# ////////////////////////////////////////////////////
# In order to made some parts of this code, I used other converors' code (specially the "onnx2pytorch" library)
# ////////////////////////////////////////////////////

# Notes:
# All the weights and the data related to the variables is stored in "model_graph.initializer" 
#   (Note: "onnx.ModelProto.graph.node" is not a node, it is a list of nodes)
# With _node being an element of onnx.ModelProto.graph.node:
#   _node.input: inputs list. _node.output: outputs list. _node.attribute: list made by all the parameteres and values (they are "AttributeProto")

def extract_shape(data: onnx.ValueInfoProto) -> np.shape:
    # The shape of the inputs/ouputs is more or less a list quite hidden.
    #   Note: ONNX allows to have shapes of undefined value, for example: (N, 3, 224, 224), 
    #       and, if it is not defined, that dimension is stored as 0. I will assume that every loaded model has declared all theirs values.
    # TODO: Mirar qué hacer en caso de que no se haya definido alguna dimensión ==> 
    #   ==> Puesto que entiendo que solo son entradas y salidas, se podría pasar como parámetro
    #   [==>] Alternativamente, como, por lo que he visto hasta ahora, son más el número de entradas/salidas que van a haber, saltarlas.
    #   ==> TODO: cuando todo esté más o menos claro, preguntárselo a Manel 
    #   (En cualquier caso, tenerlo en cuenta para la conversión en el sentido opuesto)
    return tuple([elem.dim_value for elem in data.type.tensor_type.shape.dim if elem.dim_value != 0])
# --- extract_shape --- #

def get_relevant_data(model_graph:onnx.GraphProto) -> Tuple[Dict[str, np.shape], Dict[str, np.shape], Dict[str, np.ndarray]]:
    
    # onnx.numpy_helper.to_array() is a function that transforms onnx data into a ndarray (numpy's array)
    
    # Weights dicionary. Key: weight name. Value: the onnx tensor in a numpy format (with -technically- the correct dtype).
    weights_dict = {node.name: onnx.numpy_helper.to_array(node) for node in model_graph.initializer}

    # Inputs dicionary. Key: input name. Value: the shape of the input.
    inputs_dict = {_input.name: extract_shape(_input) 
                    for _input in model_graph.input if _input.name not in weights_dict.keys()}
    
    # Outputs dicionary. Key: output name. Value: the shape of the output.
    outputs_dict = {ouput.name: extract_shape(ouput) for ouput in model_graph.output}

    return (inputs_dict, outputs_dict, weights_dict)
# --- END get_inputs_outputs_and_attributes_names --- # 


def extract_attributes(node: onnx.NodeProto) -> Dict[str, Any]:

    return {attribute.name: onnx.helper.get_node_attr_value(node, attribute.name) 
            for attribute in node.attribute}
# --- END extract_attributes --- #

def get_operations(onnx_model:onnx.ModelProto, opset_version:int, inputs: Dict[str, np.shape], 
                   outputs: Dict[str, np.shape]) -> List[LayerAndActivationBase]:

    # TODO: meter otros parámetros que se puedan necesitar
    #operations = list()
    #for node in onnx_model.graph.node:
    #    parameters = [weights[par_name] for par_name in node.input if par_name in weights]
    #    operations.append(SWITCH_ONNX_TO_PYDTNN[node](node, parameters))

    # TODO: implementar las funciones del "Switch"
    
    operations = [Input(shape=inputs)]
    for i in range( len(onnx_model.graph.node) - 1 ):
        node = onnx_model.graph.node[i]
        info = {CONST_NODE : node, CONST_OPSET : opset_version,
                 CONST_ATTRIBUTES: extract_attributes(node=node)}
        operations.append(SWITCH_OPERATION_ONNX_TO_PYDTNN[node.name](info))

    # It is assumed that the last layer always has the "shape" attribute and that is relevant in order to make the network's output.
    node = onnx_model.graph.node[-1]
    info = {CONST_NODE : node, CONST_OPSET : opset_version, 
            CONST_OUPTUS: outputs, CONST_ATTRIBUTES: extract_attributes(node=node)}
    operations.append(SWITCH_OPERATION_ONNX_TO_PYDTNN[node.name](info))

    return operations
# --- END get_operations --- #

def load_layers(model:PyDTNN_Model, operations:List[LayerAndActivationBase]) -> None:
    
    for operation in operations:
        # This is done this way because sometimes a ONNX operation can be a list of PyDTNN operations.
        #if not isinstance(operation, list):
        #    operation = [operation]
        #else: Nothing special.

        # Actually adding the layers to the model.
        for op in operation:
            model.add(op)

    return # None (No value is returned)
# --- END load_layers --- #

def convert_model(onnx_model:onnx.ModelProto, omm=None, non_blocking_mpi=False, enable_gpu=False, enable_gpudirect=False,
                 enable_nccl=False, dtype=np.float32, tracing=False, tracer_output="", **kwargs) -> PyDTNN_Model:
    
    # Output model.
    model = PyDTNN_Model(omm=omm, non_blocking_mpi=non_blocking_mpi, enable_gpu=enable_gpu, enable_gpudirect=enable_gpudirect,
                 enable_nccl=enable_nccl, dtype=dtype, tracing=tracing, tracer_output=tracer_output, **kwargs)

    # Obtaining the relevant data (inputs, outputs, weights, ...) from the onnx model.
    inputs, outputs, weights = get_relevant_data(onnx_model.graph)
    opset_version = onnx_model.opset_import[0].version
    
    # Obtaining the operations (layes, activations, etc.).
    operations = get_operations(onnx_model=onnx_model, opset_version=opset_version, inputs=inputs, outputs=outputs)
    # Asigning the operations to the model.
    load_layers(model=model, operations=operations)

    # TODO: Faltaría comprobar cómo se conectan las entradas entre ellas
    # TODO: Faltaría comprobar el formato de los pesos (y hacer la traducción, si fuese necesario)

    # Loading the weights into the model.
    model.load_store_path(layers = model.layers, d = weights, mode = "load")
        
    return model
# --- END convert_model --- #
