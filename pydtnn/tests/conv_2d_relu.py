from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.initializers import glorot_uniform, zeros
from pydtnn.utils.tensor import TensorFormat
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.tests.abstract.common import Params
from pydtnn.tests.abstract.common import D
from pydtnn.model import Model
from pydtnn.backends.fuse.layers.conv_2d_relu import Conv2DRelu
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.activations.relu import Relu
import unittest
import logging
logger = logging.getLogger(__name__)


# TODO: Mirar esto.


class Conv2DReluTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D+Relu leads to the same results than Conv2DRelu
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[ConcatenationBlock, AbstractConv2D]:
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b  # type: ignore (it's okay)
        params.backend = "cpu;conv_2d:gemm"
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride),
                        dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)

        relu = Relu()
        chain = ConcatenationBlock([
            conv2d,
            relu
        ])
        shape = (d.c, d.h, d.w)
        chain._init_backend_with_model(model)
        chain._model_init(prev_shape=shape, x=None)

        from_parent = (relu.__dict__ | conv2d.__dict__)
        fuse = Conv2DRelu(from_parent=from_parent)
        fuse.init_backend_with_model(model)
        fuse.__dict__.update(from_parent)
        fuse.initialize(prev_shape=shape, x=None)

        # Set the same initial weights and biases to both layers
        fuse.weights = conv2d.weights.copy()
        fuse.biases = conv2d.biases.copy()

        return chain, fuse

    @staticmethod
    def _set_state(layer: Conv2D, weights) -> None:
        if isinstance(layer, ConcatenationBlock):
            layer.paths[0][0].weights = weights.copy()
        else:
            layer.weights = weights.copy()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride3(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_larger_handmade_array_stride2(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_handmade_array(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3_filter1x2(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3_filter1x1(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_defaults(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_handmade_array_stride2(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride3(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_even_larger_handmade_array_stride12(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_alexnet_imagenet_first_conv2d(self):
        raise NotImplementedError()

    @unittest.skip("Backward not implemented")
    def test_forward_backward_alexnet_cifar10_first_conv2d(self):
        raise NotImplementedError()
