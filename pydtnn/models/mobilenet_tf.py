from pydtnn.layers import *
from pydtnn.activations import *
from pydtnn.layers.conv_2d import GroupingEnum

# NOTE: TensorFlow uses AveragePool2D with (2, 2) pool shape

def mobileNetTF(input_shape, output_shape):
    """
    Mobilenet v1's TF.Keras-like version.
    """

    first_filters = 32

    epsilon = 1e-3
    momentum = 0.99

    model = []
    _ = model.append
    _( Input(shape=input_shape) )
    _( Conv2D(nfilters=first_filters, filter_shape=(3,3), grouping=GroupingEnum.STANDARD, padding=0, stride=2, use_bias=False) )
    _( BatchNormalization(epsilon=epsilon, momentum=momentum) )
    _( Relu6() )

    layout = [ [64, 1], [128, 2], [256, 2], [512, 6], [1024, 2] ]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _( Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=GroupingEnum.DEPTHWISE, padding=1, stride=stride, use_bias=False) )
            _( BatchNormalization(epsilon=epsilon, momentum=momentum) )
            _( Relu6() )
            _( Conv2D(nfilters=n_filt, filter_shape=(1, 1), grouping=GroupingEnum.POINTWISE, use_bias=False) )
            _( BatchNormalization(epsilon=epsilon, momentum=momentum) )
            _( Relu6() )
            first_filters = n_filt

    _( AveragePool2D(pool_shape=(1, 1)) )
    _( Flatten() )
    _( FC(shape=(512,), activation = relu) )
    _( Dropout(0.3) )
    _( FC(shape=output_shape, activation=softmax) )

    return model

create_mobilenet_tf = mobileNetTF