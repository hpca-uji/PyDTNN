**constants.py**

In this script there are:
- Some constants that act as a keys for some dictionaries (if this were written in C, this would be "#define"s)
- A common operation that transforms the ONNX's padding format to the PyDTNN's one.
- And a pseudo-switch to transoform ONNX operations into their PyDTNN's equivalents. It's a dictionary where the key is a string with the name of the ONNX operation, and the value is a function where the "translation" will be made (if it is implemented).

**model_convertor.py**

Here is all the logic that extracts the data from the ONNX graph and rebuild the model as the PyDTNN's equivalent.

The function are: 
-extract_shape: This function extract the shape of the graph inputs and outputs.

-get_relevant_data: This function extract from the ONNX graph the model's inputs (only 1st layer), outputs (only last layer) and weights (all the model).

-extract_attributes: This function extracts the ONNX operation attributes (PyDTNN layer parameters) from a ONNX node as a numpy array.

-get_lists_operations_and_outputs: This function is only called when a operation has more than one input (e.g.: "Concat", operation that concatenate the results of two operations). It prepare the data to make the convertion.
[NOTE: due the way PyDTNN work with "Add" and "Concat" layers, in order convert theese operations in a "easy" way, all the operations are stored in a dictionary like {[operation_output]: ([list of operations], [inputs]) }]

-get_actual_inputs: in ONNX, a operation has as a input the previous layer (or layers) output and the weights and biases. This function gets the lists of inptuts (without the weigths, biases and other parameters).

-get_layers: this function (along with "_get_and_put_layer") uses the "constants.py"'s "pseudo-switch" in order to make the convertion.

-load_layers: This function add all the ONNX's converted layers into the PyDTNN's model (basically, it does the model.add. It is in a different function in case it would be necessary to do more operations; if not, it can be a good idea to put this logic into "convert_model").

-convert_model: The "main" function of this translation. It call the functions above in order to make the convertion, and returns (if it would work) a PyDTNN equivalent model.

**./operations/* **

In this folder there are the scripts where the ONNX's convertion operations are implemented.
Due ONNX has more than 200 operations, they are separated in files by their initial character (e.g: 'Add', 'AveragePool' are in "A.py", and 'Relu' is in "R.py"). Also, there are some special operations that has their own script.
Only are implemented (more or less): 'Add', 'AveragePool', 'BatchNormalization', 'Concat', 'Conv', 'Dropout', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Mul', 'Relu', 'Unsqueeze'
NOTE: Check GEMM (the most similar operation is FC, but it's not equivalent) and Unsqueeze (there is no equivalent in PyDTNN).

