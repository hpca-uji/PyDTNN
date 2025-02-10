# Typing related
from typing import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
import numpy as np

# Operations/transformations related
import onnx
from model import Model as PyDTNN_Model
from constants import SWITCH_OPERATION_ONNX_TO_PYDTNN, CONST_NODE, CONST_OPSET, CONST_OUPTUS, CONST_ATTRIBUTES, CONST_INPUTS, CONST_LISTS_NODES
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

def get_lists_operations(info: Dict[str, Any], operations: Dict[str, Tuple[LayerAndActivationBase, List[str] ]])-> List[List[LayerAndActivationBase]]:

    # NOTE: It is assumed that the model will by a feed-forward netowork 
    dict_branch = {}

    # Making the "path" of layers for every input
    for inpt in info[CONST_INPUTS]:
        dict_branch[inpt] = dict() 
        
        input_search = inpt
        while input_search is not None:
            #operations: {[output_name]: ([operation], [inputs])}
            op, inp = operations[input_search]
            if inp is None:
                # case: root layer.                
                dict_branch[inpt][input_search] = op
                input_search = None
            else:
                input_search = inp[0] # The inputs list should have only one input.
                dict_branch[inpt][input_search] = op

    # Searching the first coincidence

    # Sets are not ordered by insertion ==> keep order with enumerate ==>
    #   ==> braches have different sizes, then the same node may have different order in different branches ==> 
    #   ==> that's true from bottom to top, from top to bottom the "intersection layers" (the ones to be searched) should have the same position.
    enumerated_reversed_inputs = enumerate(list(dict_branch[info[CONST_INPUTS][0]].keys())[::-1])

    coincidences = set(enumerated_reversed_inputs)
    for i in range(1, len(info[CONST_INPUTS])):
        coincidences.intersection(set(enumerate(list(dict_branch[info[CONST_INPUTS][i]].keys())[::-1])))

    # "Unenumerating" and sorting the intersection and getting the first coincidence.
    #   ==> NOTE: Due the list was sorting in reverse before, now it is necessary to sort it be reverse again (that's why the "-x[0]").
    coincidence = [elem[1] for elem in sorted(coincidences, key=lambda x: -x[0])][0]

    # Trimming the lists from that element (first coincidence)
    lists_operations = list()
    for inpt in info[CONST_INPUTS]:
        _values = list(dict_branch[inpt].values())
        lists_operations.append(_values[:_values.index(operations[coincidence][0])])

    return lists_operations

# --- END get_lists_operations --- #


def get_actual_inputs(list_inputs: List[str], weights_names: List[str])-> List[str]:
    # This function' objective is to remove non layer-to-layer onnx inputs (e.g.: the weigth [_weight], the bias [_bias], etc. ).
    #   To do that, only the inputs that end with the accepted ending remains.
    return list(filter(lambda _input: _input not in weights_names, list_inputs))
# --- END get_actual_inputs --- #

def _get_and_put_operation(node: onnx.NodeProto, opset_version:int, operations: Dict[str, Tuple[LayerAndActivationBase, List[str]]], output: str|None = None)->None:
        info = {CONST_NODE : node, # Refererence to the model itself (TODO: see if it's necessary. If not ==> delete)
                CONST_OPSET : opset_version,    # Version of the onnx operation
                CONST_INPUTS: get_actual_inputs(node.input),   # node's inputs names
                CONST_OUPTUS: node.output if output is None else output,  # node's outputs names or the model's output (TODO: Check if a operation can have multiple outputs)
                CONST_ATTRIBUTES: extract_attributes(node=node) # dictionary with the node's attributes names and respective values (e.g. the shape of a kernel)
                }
        if len(info[CONST_INPUTS]) > 1:
            info[CONST_LISTS_NODES] = get_lists_operations(info, operations)
        operations[info[CONST_OUPTUS]] = tuple(SWITCH_OPERATION_ONNX_TO_PYDTNN[node.name](info), info[CONST_INPUTS])

    # return Nothing: the output is stored in the dictionary
# --- END _get_and_put_operation --- #

def get_operations(onnx_model:onnx.ModelProto, opset_version:int, inputs: Dict[str, np.shape], 
                   outputs: Dict[str, np.shape]) -> List[LayerAndActivationBase]:

    # TODO: meter otros parámetros que se puedan necesitar
    #operations = list()
    #for node in onnx_model.graph.node:
    #    parameters = [weights[par_name] for par_name in node.input if par_name in weights]
    #    operations.append(SWITCH_ONNX_TO_PYDTNN[node](node, parameters))

    # TODO: implementar las funciones necesarias del "Switch"
    # TODO: Hay outputs que se pueden pasar como inputs capas posteriores ==> Mirar cómo conectar las cosas.
    #   ==> Tal vez hacer un diccionario de [nombres de salidas, capa] para que, cuando una capa tenga como entrada ese nombre, pueda relacionarlo rápidamente.
    
    # Si una operación tiene "n" entradas hay que (asumiendo que no hay redes recursivas):
    # - Identificar en qué punto se hacen las n separaciones (el nodo raíz que se divide en "n" ramas)
    # -> Para esto:
    # ==> Para cada entrada:
    #   ==> Se accede en el diccionario a la entrada de el input para formar la ruta desde el nodo input del add hasta llegar al nodo raíz.
    # ==> Una vez se tienen ambas ramas, desde el add hasta el nodo raíz, se mira cual es el primer inptu que coincide (esto indica donde se separan las ramas).
    # - Guardar cada rama en una lista y todas las listas en una lista de listas para pasar a la capa (creo que esto solo lo permite operaciones como "add")    
    # -> Tras eso:
    # ==> Para cada nodo, habrá que hacer una lista con los nodos entre el que tiene varias entradas y el que tiene una salida que va a varios nodos (sin incluirlo)
    # ==> Asumiendo que el Add es la única capa que tiene entradas múltiples (o que todas siguen su formato), hay que pasarlas como el parámetro y eliminarlas de la lista de operaciones que añadir (ya están añadidas implícitamente)
    
    # It is expected to have at least one layer.
    num_operations = len(onnx_model.graph.node)
    assert num_operations > 0

    # operations: {[output_name]: ([operation], [inputs])}
    output_first_layer = get_actual_inputs(onnx_model.graph.node[0].input)
    operations = {output_first_layer : (Input(shape=inputs), [None])}

    for i in range(num_operations - 1):
        _get_and_put_operation(node=onnx_model.graph.node[i], opset_version=opset_version, operations=operations)
    _get_and_put_operation(node=onnx_model.graph.node[-1], opset_version=opset_version, operations=operations, output=outputs)

    # The list of layers is returned.
    return list(map(lambda x: x[0], operations.values()))
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
