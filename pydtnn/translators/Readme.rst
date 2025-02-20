Both onnx2pydtnn and pydtnn2onnx are work in progress.

[--------]
**onnx2pydtnn**: 
- It's possible to more get an equivalent PyDTNN' layer for some Onnx' operations, that are the operations necessary to make DenseNet169, VGG19 and ResNet50:
-> 'Add', 'AveragePool', 'BatchNormalization', 'Concat', 'Conv', 'Dropout', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Mul', 'Relu', 'Unsqueeze'
Due ONNX has more than 200 operations, they are separated in files by their initial character (e.g: 'Add', 'AveragePool' are in "A.py", and 'Relu' is in "R.py").
Also, there are some special operations that has their own script (but they aren't implemented).

Problem:
- As it is possible to see at the end of the file "output_test1_2" (translators/onnx2pydtnn/test/output_test1_2), the construction fails. This can be produced because, in at least one layer, some attributes aren't being translated in a proper way.

**pydtnn2onnx**: 
- Basically, it doesn't works. There are only some functions with some work, but not too much.

[--------]
Interesting tool: https://netron.app/