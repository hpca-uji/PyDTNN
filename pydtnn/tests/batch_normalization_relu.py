import unittest

from pydtnn.activations.relu import Relu
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.batch_normalization_relu import BatchNormalizationRelu
from pydtnn.model import Model
from pydtnn.tests.abstract.common import D, Params
from pydtnn.tests.abstract.conv_2d_common import Conv2DCommonTestCase
from pydtnn.utils.tensor import TensorFormat


class BatchNormalizationReluTestCase(Conv2DCommonTestCase):
    """
    Tests that BatchNormalization+Relu leads to the same results than BatchNormalizationRelu
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global Conv2DCommonTestCase
    del Conv2DCommonTestCase

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple:
        params = Params()
        params.tensor_format = TensorFormat.NCHW.upper()
        params.batch_size = d.b
        params.conv_variant = "gemm"
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        bn = BatchNormalization()
        relu = Relu()
        chain = ConcatenationBlock([
            bn,
            relu
        ])
        shape = (d.c, d.h, d.w)
        chain.init_backend_from_model(model)
        chain.initialize(prev_shape=shape, x=None)

        from_parent = relu.__dict__ | bn.__dict__
        fuse = BatchNormalizationRelu(from_parent=from_parent)
        fuse.init_backend_from_model(model)
        fuse.__dict__.update(from_parent)
        fuse.initialize(prev_shape=shape, x=None)

        # Set the same initial weights and biases to both layers
        fuse.running_mean = bn.running_mean.copy()
        fuse.running_var = bn.running_var.copy()

        return chain, fuse

    @staticmethod
    def _set_state(layer: BatchNormalization, weights) -> None:
        pass

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
