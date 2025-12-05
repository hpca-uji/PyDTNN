import unittest

from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_batch_normalization import Conv2DBatchNormalization
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.conv_2d import Conv2DCPU
from pydtnn.tests.abstract.common import D
from pydtnn.tests.abstract.common import Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils.initializers import glorot_uniform, zeros


class Conv2DBatchNormalizationTestCase(Conv2DCommonTestCase):
    """
    Tests that Conv2D+BatchNormalization leads to the same results than Conv2DBatchNormalization
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[Conv2DCPU, Conv2DCPU]:
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b
        params.conv_variant = "gemm"
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride),
                        dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        bn = BatchNormalization()
        chain = ConcatenationBlock([
            conv2d,
            bn
        ])
        shape = (d.c, d.h, d.w)
        chain.init_backend_from_model(model)
        chain.initialize(prev_shape=shape, x=None)

        from_parent = bn.__dict__ | conv2d.__dict__
        fuse = Conv2DBatchNormalization(from_parent=from_parent)
        fuse.init_backend_from_model(model)
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
