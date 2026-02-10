from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.model import Model
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.tests.abstract.common import D
from pydtnn.tests.abstract.common import Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.initializers import glorot_uniform, zeros


class Conv2DGroupTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D with Depth+Pair leads to the same results than Conv2D Standard
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[AbstractConv2DNumpy, AbstractConv2DNumpy]:
        params = Params()
        params.tensor_format = TensorFormat.NHWC.upper()
        params.batch_size = d.b
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d_depth = Conv2DDepthwise(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                                       padding=(d.vpadding, d.hpadding),
                                       stride=(d.vstride, d.hstride),
                                       dilation=(d.vdilation, d.hdilation),
                                       use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d_pair = Conv2DPointwise(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                                      padding=(d.vpadding, d.hpadding),
                                      stride=(d.vstride, d.hstride),
                                      dilation=(d.vdilation, d.hdilation),
                                      use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        chain = ConcatenationBlock([
            conv2d_depth,
            conv2d_pair
        ])
        chain._init_backend_with_model(model)
        chain._model_init(prev_shape=(d.c, d.h, d.w), x=None)

        conv2d = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride),
                        dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d._init_backend_with_model(model)
        conv2d._model_init(prev_shape=(d.c, d.h, d.w), x=None)

        # Set the same initial weights and biases to both layers
        conv2d_depth.weights = conv2d.weights.copy()
        conv2d_depth.biases = conv2d.biases.copy()
        conv2d_pair.weights = conv2d.weights.copy()
        conv2d_pair.biases = conv2d.biases.copy()

        return conv2d, chain

    @staticmethod
    def _set_state(layer: Conv2D, weights) -> None:
        if isinstance(layer, ConcatenationBlock):
            layer.paths[0][0].weights = weights.copy()
            layer.paths[0][1].weights = weights.copy()
        else:
            layer.weights = weights.copy()
