# Typing-related imports
from typing import *
from inspect import stack # This is only in order to get the function's name

# Funtionality imports
from onnx import NodeProto
from onnx.helper import make_node
import constants as cons

# This function makes an activation node without attributes.
#   NOTE: in case there are attributes, make a function for that operation.
def make_activation_node(info:dict[str, Any]) -> NodeProto:    
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    
    op_name = info[cons.CONST_OP_NAME]
    num_operation = info[cons.CONST_NUM_OP]
    inputs = info[cons.CONST_INPUTS]
    outputs = cons.make_output_name(op_name, num_operation)

    return make_node(op_type=op_name, inputs=inputs, outputs=outputs, name=outputs, 
                     doc_string="Read https://onnx.ai/onnx/operators/ and https://github.com/hpca-uji/PyDTNN for more information.",
                     domain="main"
                     )
# --- END make_activation_node --- #
    
def Softmax(info:dict[str, Any])-> NodeProto:
    print(f"Operation: {stack()[0].function}\nargs received: {info}")
    
    op_name = info[cons.CONST_OP_NAME]
    num_operation = info[cons.CONST_NUM_OP]
    inputs = info[cons.CONST_INPUTS]
    outputs = cons.make_output_name(op_name, num_operation)

    #  Value: pydtnn.activations.softmax
    attribute = {"axis": 1}

    return make_node(op_type="Softmax", 
                     inputs=inputs, outputs=outputs, name=outputs, 
                     doc_string="Read https://onnx.ai/onnx/operators/onnx__Softmax.html and https://github.com/hpca-uji/PyDTNN for more information.",
                     kwargs=attribute)
# --- END Softmax --- #

