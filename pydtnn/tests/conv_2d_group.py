from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.conv_2d import Conv2DCPU
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
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[Conv2DCPU, Conv2DCPU]:
        params = Params()
        params.tensor_format = TensorFormat.NHWC.upper()
        params.batch_size = d.b
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d_depth = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                              grouping=Conv2D.Grouping.DEPTHWISE,
                              padding=(d.vpadding, d.hpadding),
                              stride=(d.vstride, d.hstride),
                              dilation=(d.vdilation, d.hdilation),
                              use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d_pair = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                             grouping=Conv2D.Grouping.POINTWISE,
                             padding=(d.vpadding, d.hpadding),
                             stride=(d.vstride, d.hstride),
                             dilation=(d.vdilation, d.hdilation),
                             use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        chain = ConcatenationBlock([
            conv2d_depth,
            conv2d_pair
        ])
        chain.init_backend_from_model(model)
        chain.initialize(prev_shape=(d.c, d.h, d.w), x=None)

        conv2d = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        grouping=Conv2D.Grouping.STANDARD,
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride),
                        dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d.init_backend_from_model(model)
        conv2d.initialize(prev_shape=(d.c, d.h, d.w), x=None)

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
