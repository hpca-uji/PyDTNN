# Typing related
from typing import Any
from pydtnn.layer_base import LayerBase
import numpy as np

# Operations/transformations related
import onnx
from pydtnn.model import Model as PyDTNN_Model
import pydtnn.converters.pydtnn2onnx.constants as cons
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


def get_relevant_data(model_graph: onnx.GraphProto) -> tuple[dict[str, np.shape], dict[str, np.shape], dict[str, np.ndarray]]:

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


def extract_attributes(node: onnx.NodeProto) -> dict[str, Any]:

    return {attribute.name: onnx.helper.get_node_attr_value(node, attribute.name)
            for attribute in node.attribute}
# --- END extract_attributes --- #


def get_lists_operations_and_outputs(info: dict[str, Any], operations: dict[str, tuple[LayerBase, list[str]]]) -> tuple[list[list[LayerBase]], list[str]]:

    # NOTE: It is assumed that the model will by a feed-forward netowork
    dict_branch = {}

    # Making the "path" of layers for every input
    for inpt in info[cons.CONST_INPUTS]:
        dict_branch[inpt] = dict()

        input_search = inpt
        while input_search is not None:
            # operations: {[output_name]: ([operation], [inputs])}
            op, inp = operations[input_search]
            output = input_search

            if inp is None:
                # case: root layer.
                dict_branch[inpt][input_search] = (op, output)
                input_search = None
            else:
                input_search = inp[0]  # The inputs list should have only one input.
                dict_branch[inpt][input_search] = (op, output)

    # Searching the first coincidence

    # Sets are not ordered by insertion ==> keep order with enumerate ==>
    #   ==> braches have different sizes, then the same node may have different order in different branches ==>
    #   ==> that's true from bottom to top, from top to bottom the "intersection layers" (the ones to be searched) should have the same position.
    enumerated_reversed_inputs = enumerate(list(dict_branch[info[cons.CONST_INPUTS][0]].keys())[::-1])

    coincidences = set(enumerated_reversed_inputs)
    for i in range(1, len(info[cons.CONST_INPUTS])):
        coincidences.intersection(set(enumerate(list(dict_branch[info[cons.CONST_INPUTS][i]].keys())[::-1])))

    # "Unenumerating" and sorting the intersection, and getting the first coincidence layer.
    #   ==> NOTE: Due the list was sorting in reverse before, now it is necessary to sort it be reverse again (that's why the "-x[0]").
    coincidence = [elem[1] for elem in sorted(coincidences, key=lambda x: -x[0])][0]

    # Trimming the lists from that element (first coincidence)
    lists_operations = list()
    lists_outputs = list()
    for inpt in info[cons.CONST_INPUTS]:
        _values = list(dict_branch[inpt].values())
        layers = [elem[0] for elem in _values]
        outputs = [elem[1] for elem in _values]

        trimming_index = (layers.index(operations[coincidence][0]))
        lists_operations.append(layers[:trimming_index])  # Remember: list of lists
        lists_outputs.extend(outputs[:trimming_index])  # Remember: list of string

    return (lists_operations, lists_outputs)

# --- END get_lists_operations_and_outputs --- #


def get_actual_inputs(list_inputs: list[str], weights_names: list[str]) -> list[str]:
    # This function' objective is to remove non layer-to-layer onnx inputs (e.g.: the weigth [_weight], the bias [_bias], etc. ).
    #   To do that, only the inputs that end with the accepted ending remains.
    return list(filter(lambda _input: _input not in weights_names, list_inputs))
# --- END get_actual_inputs --- #


def _get_and_put_operation(node: onnx.NodeProto, opset_version: int, operations: dict[str, tuple[LayerBase, list[str]]],
                           weights: dict[str, np.ndarray], output: list[str] | None = None) -> None:

    info = {  # cons.CONST_NODE: node, # Refererence to the model itself (TODO: see if it's necessary. If not ==> delete)
        cons.CONST_OPSET: opset_version,    # Version of the onnx operation
        cons.CONST_INPUTS: get_actual_inputs(list_inputs=node.input, weights_names=list(weights.keys())),   # node's inputs names
        cons.CONST_ALL_INPUTS: node.input,   # ALL node's inputs names (including weights and biases)
        cons.CONST_OUPTUS: node.output if output is None else output,  # node's outputs names or the model's output (TODO: Check if a operation can have multiple outputs)
        cons.CONST_ATTRIBUTES: extract_attributes(node=node),  # dictionary with the node's attributes names and respective values (e.g. the shape of a kernel)
        cons.CONST_WEIGHTS: weights,
        cons.CONST_PREV_LAYERS: operations
    }
    if len(info[cons.CONST_INPUTS]) > 1:
        info[cons.CONST_listS_NODES], operations_to_remove = get_lists_operations_and_outputs(info, operations)
        # Due the way the add/concatention layer works, those must to be removed from the operations (they will be inside the other operation)
        for operation in operations_to_remove:
            del operations[operation]

    operations[info[cons.CONST_OUPTUS][0]] = (cons.switch_onnx_operation_to_pydtnn[node.op_type](info), info[cons.CONST_INPUTS])

    # return Nothing: the output is stored in the dictionary
# --- END _get_and_put_operation --- #


def get_operations(onnx_model: onnx.ModelProto, opset_version: int, inputs: dict[str, np.shape],
                   weights: dict[str, np.ndarray], outputs: dict[str, np.shape]) -> list[LayerBase]:

    # TODO: meter otros parámetros que se puedan necesitar
    # operations = list()
    # for node in onnx_model.graph.node:
    #    parameters = [weights[par_name] for par_name in node.input if par_name in weights]
    #    operations.append(SWITCH_ONNX_TO_PYDTNN[node](node, parameters))

    # TODO: implementar las funciones necesarias del "Switch"

    # It is expected to have at least one layer.
    num_operations = len(onnx_model.graph.node)
    assert num_operations > 0

    # operations: {[output_name]: ([operation], [inputs])}
    output_first_layer = get_actual_inputs(list_inputs=onnx_model.graph.node[0].input, weights_names=list(weights.keys()))[0]
    operations = {output_first_layer: (Input(shape=inputs), [None])}

    for i in range(num_operations - 1):
        _get_and_put_operation(node=onnx_model.graph.node[i], opset_version=opset_version, operations=operations, weights=weights)
    _get_and_put_operation(node=onnx_model.graph.node[-1], opset_version=opset_version, operations=operations, weights=weights, output=list(outputs.keys()))

    # The list of layers is returned.
    return list(map(lambda x: x[0], operations.values()))
# --- END get_operations --- #


def get_layers(pydtnn_model: PyDTNN_Model) -> list[onnx.NodeProto]:

    node_list = list()
    for layer in pydtnn_model.layers:

        name, number_layer = cons.get_layer_name_and_id(str(layer))
        info = {
            cons.CONST_OP_NAME: name,
            cons.CONST_NUM_OP: number_layer,
        }

        node = cons.SWITCH_OPERATION_PYDTNN_TO_ONNX[name](info)

        node_list.append(node)

    return node_list
# --- END get_nodes --- #


def convert_model(pydtnn_model: PyDTNN_Model, ir_version: int = 1, producer_name: str = "PyDTNN",
                  producer_version: str = "", domain: str = "", model_version: int = "",
                  doc_string: str = "https://github.com/hpca-uji/PyDTNN",
                  ) -> onnx.ModelProto:

    nodes = get_layers(pydtnn_model)

    # TODO:
    model = onnx.helper.make_model()

    return model
# --- END convert_model --- #
