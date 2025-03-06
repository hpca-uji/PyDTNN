Ejemplo clave pesos:
"module.features.conv0.weight"
- Si tiene "module." indica que el modelo está "recubierto" por DistributedDataParallel ==> "features.conv0.weight" sería la clave del peso de forma normal.
- "features.conv0" indica que:  
    a) features es un atributo de la clase
    b) es la capa conv0 dentro de un nn.Sequential (o algún bloque similar)

# *------------------*

import torch.fx ==> Herramienta de debug.

Con "torch.fx.symbolic_trace(_modelo_).graph" obtienes cómo quedaría la función "forward" totalemente extendida, además de cómo se relaciona cada capa entre ellas y otras llamadas a funciones
Ejemplo:
'''
    features_conv0 = self.features.conv0(x);  x = None
    features_norm0 = self.features.norm0(features_conv0);  features_conv0 = None
    features_relu0 = self.features.relu0(features_norm0);  features_norm0 = None
    features_pool0 = self.features.pool0(features_relu0);  features_relu0 = None
    cat = torch.cat([features_pool0], 1)
    features_denseblock1_denselayer1_norm1 = self.features.denseblock1.denselayer1.norm1(cat);  cat = None
    features_denseblock1_denselayer1_relu1 = self.features.denseblock1.denselayer1.relu1(features_denseblock1_denselayer1_norm1);  features_denseblock1_denselayer1_norm1 = None
    features_denseblock1_denselayer1_conv1 = self.features.denseblock1.denselayer1.conv1(features_denseblock1_denselayer1_relu1);  features_denseblock1_denselayer1_relu1 = None
    features_denseblock1_denselayer1_norm2 = self.features.denseblock1.denselayer1.norm2(features_denseblock1_denselayer1_conv1);  features_denseblock1_denselayer1_conv1 = None
    features_denseblock1_denselayer1_relu2 = self.features.denseblock1.denselayer1.relu2(features_denseblock1_denselayer1_norm2);  features_denseblock1_denselayer1_norm2 = None
    features_denseblock1_denselayer1_conv2 = self.features.denseblock1.denselayer1.conv2(features_denseblock1_denselayer1_relu2);  features_denseblock1_denselayer1_relu2 = None
    cat_1 = torch.cat([features_pool0, features_denseblock1_denselayer1_conv2], 1)
'''
Entiendo que tras el ";", sería el valor que tiene la variable.

Tras hacer varias pruebas poniéndole alias a los imports (ejemplo: from torch import cat as manolo), parece que el resultado no varía (queda como "torch.cat()", en vez de "manolo()")

# *------------------*

for name, mod in modelo.named_modules():
    print(f"{name}: {mod}")
    print("==============================")
--------------------
: DenseNet(
  (features): Sequential(
    (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu0): ReLU(inplace=True)
    (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (denseblock1): _DenseBlock(
      (denselayer1): _DenseLayer(
        (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu1): ReLU(inplace=True)
        (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu2): ReLU(inplace=True)
        (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      )
    ...
--------------------
De esta forma se puede sacar todas las capas (nn.Module) del modelo, incluidas aquellas que se contienen otras.

# *------------------*

for name, mod in modelo.named_children():
    print(f"{name}: {mod}")

features: Sequential(
  (conv0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu0): ReLU(inplace=True)
  (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (denseblock1): _DenseBlock(
    (denselayer1): _DenseLayer(
      (norm1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU(inplace=True)
      (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (norm2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu2): ReLU(inplace=True)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
    ...
)
classifier: Linear(in_features=1664, out_features=1000, bias=True)
--------------------
De esta forma se puede sacar todas las capas "hijas" del modelo (nn.Module), es decir, todas las capas dentro de un contenedor (nn.Module).

Función para conseguir todas las capas y el nombre saltándose los contenedores:

'''
def function(nombre:str = "self", modelo:nn.Module = None) -> Dict[str, nn.Module]:
    def _function(nombre:str, modelo:nn.Module, d:Dict[str, nn.Module]):
        children = list(modelo.named_children())
        if len(children) > 0:
            for nom, mod in children:
                n = ".".join([nombre, nom])
                _function(n, mod, d)                
        else:
            d[nombre] = modelo            
    #----#
    d = {}
    _function(nombre, modelo, d)
    return d
'''
#----#
--------------------
Printeando información:
'''
#d_modelo = salida de la función anterior.
def print_info(d_modelo):
  for k in d_modelo.keys():
    layer = d_modelo[k]
    print(f"{k}: {layer}")
    v = vars(layer)
    for k in v.keys():
      print(f"{k}: {v[k]}")
      if k == "_parameters":
        print(f"{list(v[k].keys())}")  
    input("-----------")
'''


A partir de "vars(layer)" puedo obtener toda la información que necesito para la traducción de una función a su equivalente en PyDTNN (creo).

--------------------

Con esta función se pueden extraer el orden de las operaciones que aparecen en el forward y las relaciones entre ellas (es decir, qué salidas son las entradas de otras funciones/capas/operaciones). [Probado con: ResNet50, Densenet169, AlexNet y GoogLeNet/Inception]

'''
def extract_functions(grafo: torch.fx.GraphModule) -> Dict[str, Tuple[str, str]]:
    d = dict()
    for line in filter(lambda x: not("return" in x or "forward" in x) , filter(lambda x: len(x)!=0, [elem.lstrip(" ") for elem in grafo.code.split("\n")])):
        _line =  line.split(";")
        func = _line[0]
        func = func.split(" = ")      
        if len(func) > 2:
            # Case: When it is a call to a function with a keyword. Example: torch.concatenate([var], axis = 1)
            output_var = func.pop(0)
            func = " = ".join(func)
        else:
            output_var, func = func
        func = func.split("(")
        if len(func) > 1:
            # Normal case. Example: "features_denseblock4_denselayer29_norm1 = self.features.denseblock4.denselayer29.norm1(cat_81);  cat_81 = None"
            #   => Situation: "self.features.denseblock4.denselayer29.norm1(cat_81)"
            args = func.pop()
            args = args.replace(")", "")
            func = func[-1]
            _func = func.split(", ")
            if len(_func) > 1:
                # Case "getattr". Example: "inception3a_branch2_1_conv = getattr(self.inception3a.branch2, "1").conv(relu_4);  relu_4 = None"
                #   => Situation: func = self.inception3a.branch2, "1").conv ||| args = relu_4            
                _func[1] = _func[1].replace(")", '').replace('\"', '')
                func = ".".join(_func)
        else:
            # Operation case: "add = layer1_0_bn3 + layer1_0_downsample_1"
            #   => Situation. func= ["layer1_0_bn3 + layer1_0_downsample_1"]
            _func = func[0].split(" ")
            func = _func.pop(1) # get the operation
            args = f"[{','.join(_func)}]"
        d[output_var] = (func, args)
    return d

CASO A REVISAR:
 '    layer1_0_downsample_0 = getattr(getattr(self.layer1, "0").downsample, '
 '"0")(maxpool);  maxpool = None\n'

Mirar de dejar la zona de la función tal cual para ejecutarla (cambiando el valor de self por el del objeto)

#----
grafo = torch.fx.symbolic_trace(alexnet())
d = extract_functions(grafo)
for elem in d.keys():
    print(f"{elem}: {d[elem]}")
'''

Input (bueno, grafo.code):
grafo.code
'\n\n\ndef forward(self, x : torch.Tensor) -> torch.Tensor:\n    features_0 = getattr(self.features, "0")(x);  x = None\n    features_1 = getattr(self.features, "1")(features_0);  features_0 = None\n    features_2 = getattr(self.features, "2")(features_1);  features_1 = None\n    features_3 = getattr(self.features, "3")(features_2);  features_2 = None\n    features_4 = getattr(self.features, "4")(features_3);  features_3 = None\n    features_5 = getattr(self.features, "5")(features_4);  features_4 = None\n    features_6 = getattr(self.features, "6")(features_5);  features_5 = None\n    features_7 = getattr(self.features, "7")(features_6);  features_6 = None\n    features_8 = getattr(self.features, "8")(features_7);  features_7 = None\n    features_9 = getattr(self.features, "9")(features_8);  features_8 = None\n    features_10 = getattr(self.features, "10")(features_9);  features_9 = None\n    features_11 = getattr(self.features, "11")(features_10);  features_10 = None\n    features_12 = getattr(self.features, "12")(features_11);  features_11 = None\n    avgpool = self.avgpool(features_12);  features_12 = None\n    flatten = torch.flatten(avgpool, 1);  avgpool = None\n    classifier_0 = getattr(self.classifier, "0")(flatten);  flatten = None\n    classifier_1 = getattr(self.classifier, "1")(classifier_0);  classifier_0 = None\n    classifier_2 = getattr(self.classifier, "2")(classifier_1);  classifier_1 = None\n    classifier_3 = getattr(self.classifier, "3")(classifier_2);  classifier_2 = None\n    classifier_4 = getattr(self.classifier, "4")(classifier_3);  classifier_3 = None\n    classifier_5 = getattr(self.classifier, "5")(classifier_4);  classifier_4 = None\n    classifier_6 = getattr(self.classifier, "6")(classifier_5);  classifier_5 = None\n    return classifier_6\n    '

Output:

features_0: ('self.features.0', 'x')
features_1: ('self.features.1', 'features_0')
features_2: ('self.features.2', 'features_1')
features_3: ('self.features.3', 'features_2')
features_4: ('self.features.4', 'features_3')
features_5: ('self.features.5', 'features_4')
features_6: ('self.features.6', 'features_5')
features_7: ('self.features.7', 'features_6')
features_8: ('self.features.8', 'features_7')
features_9: ('self.features.9', 'features_8')
features_10: ('self.features.10', 'features_9')
features_11: ('self.features.11', 'features_10')
features_12: ('self.features.12', 'features_11')
avgpool: ('self.avgpool', 'features_12')
flatten: ('torch.flatten', 'avgpool, 1')
classifier_0: ('self.classifier.0', 'flatten')
classifier_1: ('self.classifier.1', 'classifier_0')
classifier_2: ('self.classifier.2', 'classifier_1')
classifier_3: ('self.classifier.3', 'classifier_2')
classifier_4: ('self.classifier.4', 'classifier_3')
classifier_5: ('self.classifier.5', 'classifier_4')
classifier_6: ('self.classifier.6', 'classifier_5')

--------------------
# PyDTNN

Activaciones PyDTNN:
- arctanh
- log
- relu
- sigmoid
- softmax
- tanh

Capas PyDTNN:
- addition_block
- average_pool_2d
- batch_normalization
- concatenation_block
- conv_2d
- dropout
- fc
- flatten
- input
- max_pool_2d
#- Capas abstractas -#
- abstract_block_layer
- abstract_pool_2d_layer
- batch_normalization_relu
- conv_2d_batch_normalization
- conv_2d_relu
- conv_2d_batch_normalization_relu
- layer_and_activation_base
- layer

*------------------*
# PyTorch

#- Capas - #
- Conv2d
- Linear
- BatchNorm2d
- ReLU
- AdaptiveAvgPool2d
- AvgPool2d
- MaxPool2d
- Dropout

#Densenet: {'Conv2d', 'Linear', 'BatchNorm2d', 'ReLU', 'AvgPool2d', 'MaxPool2d'}
#GoogLeNet: {'Conv2d', 'Linear', 'BatchNorm2d', 'AdaptiveAvgPool2d' 'MaxPool2d', 'Dropout'}
#VGG19: {'Conv2d', 'Linear', 'AdaptiveAvgPool2d', 'ReLU', 'MaxPool2d', 'Dropout'}
#ResNet50: {'Conv2d', 'Linear', 'BatchNorm2d', 'AdaptiveAvgPool2d', 'ReLU', 'MaxPool2d'}

#- Funciones -#
TODO: Sacar qué funciones se necesitan.



####################################
##########                ##########
####################################
##########                ##########
####################################
##########                ##########
####################################
>>> b
{'self.features.0': Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.1': ReLU(inplace=True), 'self.features.2': Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.3': ReLU(inplace=True), 'self.features.4': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 'self.features.5': Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.6': ReLU(inplace=True), 'self.features.7': Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.8': ReLU(inplace=True), 'self.features.9': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 'self.features.10': Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.11': ReLU(inplace=True), 'self.features.12': Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.13': ReLU(inplace=True), 'self.features.14': Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.15': ReLU(inplace=True), 'self.features.16': Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.17': ReLU(inplace=True), 'self.features.18': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 'self.features.19': Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.20': ReLU(inplace=True), 'self.features.21': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.22': ReLU(inplace=True), 'self.features.23': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.24': ReLU(inplace=True), 'self.features.25': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.26': ReLU(inplace=True), 'self.features.27': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 'self.features.28': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.29': ReLU(inplace=True), 'self.features.30': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.31': ReLU(inplace=True), 'self.features.32': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.33': ReLU(inplace=True), 'self.features.34': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 'self.features.35': ReLU(inplace=True), 'self.features.36': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), 'self.avgpool': AdaptiveAvgPool2d(output_size=(7, 7)), 'self.classifier.0': Linear(in_features=25088, out_features=4096, bias=True), 'self.classifier.1': ReLU(inplace=True), 'self.classifier.2': Dropout(p=0.5, inplace=False), 'self.classifier.3': Linear(in_features=4096, out_features=4096, bias=True), 'self.classifier.4': ReLU(inplace=True), 'self.classifier.5': Dropout(p=0.5, inplace=False), 'self.classifier.6': Linear(in_features=4096, out_features=1000, bias=True)}
>>> print_info(b)
self.features.0: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
training: True
_parameters: OrderedDict([('weight', Parameter containing:
tensor([[[[ 0.1050, -0.0507, -0.0043],
          [-0.0955,  0.0031, -0.0563],
          [ 0.0240,  0.0169, -0.0114]],

         [[ 0.0261,  0.0014,  0.0675],
          [ 0.0013,  0.0385, -0.0300],
          [-0.0441, -0.0310, -0.0300]],

         [[ 0.0652,  0.0838,  0.0478],
          [-0.0595,  0.0389,  0.0165],
          [-0.0097,  0.0480, -0.0217]]],


        [[[ 0.0156,  0.0378,  0.0400],
          [ 0.0249,  0.0552, -0.0553],
          [ 0.0438, -0.0095,  0.0160]],

         [[-0.0504, -0.0555, -0.0686],
          [-0.0369, -0.0868,  0.0833],
          [ 0.0033,  0.0941, -0.0579]],

         [[ 0.0984, -0.0095, -0.0148],
          [ 0.0162,  0.0120,  0.0129],
          [ 0.0714,  0.0797, -0.0397]]],


        [[[-0.0127, -0.0134,  0.1123],
          [ 0.0205, -0.0850,  0.0651],
          [-0.0426,  0.0553, -0.0448]],

         [[-0.0308, -0.0165, -0.0113],
          [ 0.0378,  0.0353,  0.0646],
          [-0.0628,  0.0152,  0.0438]],

         [[ 0.0234, -0.0759,  0.0776],
          [ 0.0091,  0.0702,  0.0215],
          [ 0.0147, -0.0253, -0.0619]]],


        ...,


        [[[-0.1067,  0.0536,  0.0134],
          [-0.0034, -0.0149, -0.0668],
          [-0.1123,  0.0505, -0.0227]],

         [[ 0.0211, -0.0588,  0.0361],
          [-0.0273,  0.0533,  0.0483],
          [-0.0151,  0.0283,  0.0382]],

         [[ 0.0295,  0.0125, -0.0234],
          [-0.0394,  0.0801, -0.0208],
          [ 0.0498,  0.0194,  0.0438]]],


        [[[ 0.0391,  0.1253,  0.0398],
          [ 0.0542,  0.0229, -0.0517],
          [-0.0567,  0.0268,  0.0838]],

         [[-0.0150,  0.0617,  0.0546],
          [-0.0471,  0.0424, -0.0436],
          [ 0.0389, -0.0132,  0.0318]],

         [[ 0.0553, -0.0518, -0.0293],
          [-0.0086,  0.0029, -0.1308],
          [ 0.0111, -0.0306, -0.0564]]],


        [[[ 0.1311, -0.1546,  0.0777],
          [-0.0867, -0.0172, -0.0501],
          [ 0.0271, -0.0996, -0.0428]],

         [[-0.0545,  0.0037,  0.0933],
          [ 0.0276, -0.0172,  0.0179],
          [-0.1006,  0.0444, -0.0098]],

         [[ 0.0812,  0.0113,  0.0270],
          [ 0.0419, -0.1029, -0.0278],
          [ 0.1796, -0.0787, -0.0412]]]], requires_grad=True)), ('bias', Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       requires_grad=True))])
['weight', 'bias']
_buffers: OrderedDict()
_non_persistent_buffers_set: set()
_backward_pre_hooks: OrderedDict()
_backward_hooks: OrderedDict()
_is_full_backward_hook: None
_forward_hooks: OrderedDict()
_forward_hooks_with_kwargs: OrderedDict()
_forward_hooks_always_called: OrderedDict()
_forward_pre_hooks: OrderedDict()
_forward_pre_hooks_with_kwargs: OrderedDict()
_state_dict_hooks: OrderedDict()
_state_dict_pre_hooks: OrderedDict()
_load_state_dict_pre_hooks: OrderedDict()
_load_state_dict_post_hooks: OrderedDict()
_modules: OrderedDict()
in_channels: 3
out_channels: 64
kernel_size: (3, 3)
stride: (1, 1)
padding: (1, 1)
dilation: (1, 1)
transposed: False
output_padding: (0, 0)
groups: 1
padding_mode: zeros
_reversed_padding_repeated_twice: (1, 1, 1, 1)
-----------
self.features.1: ReLU(inplace=True)
training: True
_parameters: OrderedDict()
[]
_buffers: OrderedDict()
_non_persistent_buffers_set: set()
_backward_pre_hooks: OrderedDict()
_backward_hooks: OrderedDict()
_is_full_backward_hook: None
_forward_hooks: OrderedDict()
_forward_hooks_with_kwargs: OrderedDict()
_forward_hooks_always_called: OrderedDict()
_forward_pre_hooks: OrderedDict()
_forward_pre_hooks_with_kwargs: OrderedDict()
_state_dict_hooks: OrderedDict()
_state_dict_pre_hooks: OrderedDict()
_load_state_dict_pre_hooks: OrderedDict()
_load_state_dict_post_hooks: OrderedDict()
_modules: OrderedDict()
inplace: True
-----------
self.features.2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
training: True
_parameters: OrderedDict([('weight', Parameter containing:
tensor([[[[-2.6443e-02,  5.2086e-02,  3.7147e-02],
          [-3.5130e-02,  9.5660e-03, -3.3150e-02],
          [ 1.0748e-01,  6.2257e-02, -3.2172e-03]],

         [[-3.7528e-02, -7.6142e-02,  3.8815e-02],
          [ 1.6098e-01,  9.6736e-03,  3.6849e-02],
          [ 4.7666e-02,  6.4732e-02,  7.3676e-02]],

         [[ 7.6859e-02, -3.2642e-02, -1.1286e-01],
          [ 1.6447e-02,  2.3565e-02, -7.9980e-03],
          [ 1.5169e-01, -7.2292e-02,  1.1925e-01]],

         ...,

         [[-3.8102e-02, -5.4458e-02,  4.0932e-02],
          [-4.4028e-02,  2.8575e-02,  7.0766e-03],
          [ 1.7454e-02,  1.0898e-02,  4.8580e-02]],

         [[-1.6957e-02,  6.3613e-02,  5.7329e-02],
          [ 2.4935e-02,  5.4905e-02, -5.1835e-02],
          [ 4.6941e-02, -1.6561e-01,  1.3674e-02]],

         [[ 9.6561e-02, -1.0769e-02,  6.6763e-02],
          [ 7.2081e-02, -1.0170e-02, -1.2598e-01],
          [-1.0222e-01, -9.5249e-02, -3.8940e-02]]],


        [[[-5.0249e-02, -1.2682e-02,  5.7221e-02],
          [-6.4180e-03,  3.7993e-02,  4.6556e-02],
          [-7.3506e-02,  5.9662e-02, -7.8897e-02]],

         [[ 6.0629e-02,  3.5698e-02,  8.0253e-02],
          [ 6.1598e-03,  8.8996e-02, -1.3384e-01],
          [ 5.1994e-02,  3.8337e-02, -2.6780e-02]],

         [[-1.6217e-02, -3.0620e-02,  6.4707e-02],
          [-4.1098e-02, -1.3785e-02, -3.7878e-02],
          [-1.6370e-02,  2.9546e-02, -4.1749e-02]],

         ...,

         [[ 3.5894e-02, -4.1150e-02,  4.1630e-02],
          [-1.4943e-01, -2.9543e-02,  2.0108e-02],
          [-1.2002e-02,  1.1689e-01, -1.6380e-02]],

         [[ 2.9630e-02,  7.3251e-02, -2.6228e-02],
          [-4.6784e-02, -1.0698e-02,  3.4225e-02],
          [-7.8529e-02,  2.7540e-02,  8.8494e-02]],

         [[-2.0797e-02, -6.0509e-02,  9.7375e-02],
          [-2.3564e-03,  2.7174e-02, -4.0245e-02],
          [ 6.0987e-02, -5.9401e-02, -1.0373e-01]]],


        [[[-8.0482e-02, -2.4079e-02, -5.2619e-02],
          [-7.8771e-02,  8.1568e-02,  9.9748e-02],
          [ 2.0465e-02, -3.8428e-02,  3.3099e-02]],

         [[ 1.9995e-02,  4.0556e-02,  4.1806e-03],
          [-8.3078e-02, -5.5309e-02,  5.0492e-02],
          [ 1.9504e-01,  3.9990e-02,  1.8274e-03]],

         [[-2.2665e-02, -6.0961e-02,  4.7586e-02],
          [ 1.2786e-02, -8.2996e-02, -3.7393e-02],
          [ 3.9862e-02, -3.4277e-02, -5.6218e-03]],

         ...,

         [[-2.3437e-02,  4.3154e-02, -2.6835e-02],
          [-1.2572e-02, -5.7952e-03,  4.0900e-02],
          [ 1.1849e-02, -8.6655e-02,  1.9445e-02]],

         [[ 2.1234e-02, -8.8311e-02,  3.9578e-02],
          [-2.8691e-03, -1.5816e-02, -1.3015e-02],
          [-1.4373e-02,  3.3532e-02,  2.8099e-02]],

         [[ 9.4644e-02,  5.9825e-02,  7.4072e-02],
          [ 3.3727e-02,  7.6959e-02, -1.9603e-02],
          [-2.4730e-02,  1.8507e-01,  7.4055e-02]]],


        ...,


        [[[ 2.0824e-02,  6.8946e-02, -4.9173e-02],
          [-5.4315e-03, -7.1711e-02,  1.0662e-02],
          [-3.9933e-02, -1.0420e-02, -6.3870e-02]],

         [[ 7.7712e-02, -5.6149e-02, -1.3766e-01],
          [ 4.4003e-02, -1.7323e-03, -5.2859e-02],
          [ 7.7722e-02, -5.0576e-02,  1.0356e-01]],

         [[ 2.6435e-02, -5.5708e-02,  1.6933e-02],
          [ 3.2939e-02,  1.0316e-01, -5.7348e-02],
          [-7.7681e-02,  9.9075e-02,  3.2218e-02]],

         ...,

         [[-2.1695e-02, -3.6276e-02, -2.4370e-02],
          [-1.2677e-02, -7.5020e-02,  2.9553e-02],
          [-3.2685e-02, -3.5184e-02, -2.1257e-03]],

         [[ 3.3770e-05, -8.4777e-02, -4.4737e-02],
          [-3.3158e-02, -2.4575e-02,  1.0433e-01],
          [ 2.3596e-02, -2.1291e-02,  1.4302e-03]],

         [[ 1.0544e-01, -1.2828e-02,  4.8527e-02],
          [-5.6766e-02,  4.7219e-02, -1.3905e-02],
          [-9.4883e-03, -1.4643e-02, -5.1263e-02]]],


        [[[ 1.9269e-02, -1.8566e-04, -1.4871e-02],
          [-7.1679e-03,  5.8074e-02,  4.3744e-02],
          [-1.6833e-02, -2.8458e-02, -6.6265e-02]],

         [[-1.9799e-02,  4.7117e-03,  5.9136e-02],
          [-5.0834e-02,  2.1259e-02,  6.6379e-02],
          [-1.0952e-01,  2.0004e-02, -7.4456e-02]],

         [[-6.7663e-02,  9.8470e-02,  7.9572e-02],
          [-1.1250e-02,  1.8361e-02,  3.3762e-02],
          [ 8.6787e-02, -1.3804e-03, -2.3680e-03]],

         ...,

         [[-1.2947e-02,  4.8842e-02,  9.0509e-03],
          [-7.5013e-02,  5.2198e-03, -2.4868e-02],
          [-1.1819e-02, -3.7111e-02, -1.2755e-01]],

         [[ 2.4956e-02, -3.2781e-02,  5.2662e-02],
          [-4.9999e-02,  5.3899e-02,  3.3923e-03],
          [-7.3522e-02, -3.0405e-02, -5.8911e-02]],

         [[-2.1928e-02,  2.3098e-02,  4.7375e-02],
          [ 7.5559e-02, -7.6766e-02, -1.3805e-02],
          [ 5.7155e-02,  2.5261e-02, -4.0839e-02]]],


        [[[ 3.2411e-02,  4.4674e-02,  7.2045e-03],
          [-8.6133e-02, -2.0838e-02,  3.9939e-02],
          [ 3.2229e-02, -6.2066e-02,  8.6387e-03]],

         [[ 7.6436e-02, -6.2522e-02, -2.7071e-02],
          [-6.3594e-02,  7.4654e-02,  7.5734e-02],
          [-1.4783e-02,  4.5148e-03,  8.9436e-02]],

         [[-2.1460e-02, -9.6567e-02, -1.0549e-01],
          [ 9.2228e-02, -3.5086e-02, -5.4063e-02],
          [-3.1890e-02, -6.9260e-03, -5.4516e-03]],

         ...,

         [[ 1.9747e-02,  7.2447e-02,  8.2679e-02],
          [ 5.8314e-03,  7.6418e-02, -5.0766e-02],
          [-2.2149e-02, -5.0991e-02,  1.0142e-01]],

         [[ 4.4524e-02, -4.5954e-02, -5.4773e-02],
          [-1.4504e-02, -4.5085e-02,  5.1207e-02],
          [ 1.3037e-01, -2.5246e-02, -1.3661e-02]],

         [[ 5.7711e-02,  1.0371e-01,  1.1194e-01],
          [ 3.8243e-02, -9.1102e-02, -1.6739e-03],
          [ 1.6759e-02,  1.0211e-02,  6.6986e-03]]]], requires_grad=True)), ('bias', Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       requires_grad=True))])
['weight', 'bias']
_buffers: OrderedDict()
_non_persistent_buffers_set: set()
_backward_pre_hooks: OrderedDict()
_backward_hooks: OrderedDict()
_is_full_backward_hook: None
_forward_hooks: OrderedDict()
_forward_hooks_with_kwargs: OrderedDict()
_forward_hooks_always_called: OrderedDict()
_forward_pre_hooks: OrderedDict()
_forward_pre_hooks_with_kwargs: OrderedDict()
_state_dict_hooks: OrderedDict()
_state_dict_pre_hooks: OrderedDict()
_load_state_dict_pre_hooks: OrderedDict()
_load_state_dict_post_hooks: OrderedDict()
_modules: OrderedDict()
in_channels: 64
out_channels: 64
kernel_size: (3, 3)
stride: (1, 1)
padding: (1, 1)
dilation: (1, 1)
transposed: False
output_padding: (0, 0)
groups: 1
padding_mode: zeros
_reversed_padding_repeated_twice: (1, 1, 1, 1)
-----------
self.features.3: ReLU(inplace=True)
training: True
_parameters: OrderedDict()
[]
_buffers: OrderedDict()
_non_persistent_buffers_set: set()
_backward_pre_hooks: OrderedDict()
_backward_hooks: OrderedDict()
_is_full_backward_hook: None
_forward_hooks: OrderedDict()
_forward_hooks_with_kwargs: OrderedDict()
_forward_hooks_always_called: OrderedDict()
_forward_pre_hooks: OrderedDict()
_forward_pre_hooks_with_kwargs: OrderedDict()
_state_dict_hooks: OrderedDict()
_state_dict_pre_hooks: OrderedDict()
_load_state_dict_pre_hooks: OrderedDict()
_load_state_dict_post_hooks: OrderedDict()
_modules: OrderedDict()
inplace: True
-----------
self.features.4: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
training: True
_parameters: OrderedDict()
[]
_buffers: OrderedDict()
_non_persistent_buffers_set: set()
_backward_pre_hooks: OrderedDict()
_backward_hooks: OrderedDict()
_is_full_backward_hook: None
_forward_hooks: OrderedDict()
_forward_hooks_with_kwargs: OrderedDict()
_forward_hooks_always_called: OrderedDict()
_forward_pre_hooks: OrderedDict()
_forward_pre_hooks_with_kwargs: OrderedDict()
_state_dict_hooks: OrderedDict()
_state_dict_pre_hooks: OrderedDict()
_load_state_dict_pre_hooks: OrderedDict()
_load_state_dict_post_hooks: OrderedDict()
_modules: OrderedDict()
kernel_size: 2
stride: 2
padding: 0
dilation: 1
return_indices: False
ceil_mode: False
