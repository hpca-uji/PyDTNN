"""Test suite for verifying model consistency across different tensor formats."""

import logging
import unittest

import numpy as np

from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.layers.abstract.layer import LayerError
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.model import Model
from pydtnn.tests.abstract.common import Params, verbose_test
from pydtnn.tests.abstract.model_common import ModelCommonTestCase
from pydtnn.utils.tensor import TensorFormat, format_transpose

__all__ = ("ModelTensorTestCase",)

logger = logging.getLogger(__name__)


class ModelTensorTestCase(ModelCommonTestCase):
    """Tests that two models with different parameters lead to the same results"""

    global ModelCommonTestCase

    rtol_dict = ModelCommonTestCase.rtol_dict | {
        ConcatenationBlock: 1e-0,
        AdditionBlock: 1e-1,
        Conv2D: 1e-3,
    }
    atol_dict = ModelCommonTestCase.atol_dict | {
        ConcatenationBlock: 1e-0,
        AdditionBlock: 1e-1,
        Conv2D: 1e-3,
    }

    # NOTE: Delete parent test to prevent re-export and re-testing
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using the CPU backend tensor format NHWC"
    model2_desc = "using the CPU backend tensor format NCHW"

    def get_model2(self, model_name: str) -> Model:
        """
        Instantiates a model configured for NCHW tensor format.

        Args:
            model_name: The name of the model to instantiate.

        Returns:
            Model: The initialized model instance.
        """

        # Tensor format NCHW
        params = Params()
        params.model_name = model_name
        params.tensor_format = TensorFormat.NCHW.upper()
        params_dict = vars(params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(
                f"Model {model_name} incompatible with {params_dict['dataset_name']}"
            ) from exc
        model2._model_init()
        return model2

    @staticmethod
    def nhwc2nchw(x: np.ndarray) -> np.ndarray:
        """
        Transposes a tensor from NHWC to NCHW format.

        Args:
            x: Input numpy array in NHWC format.

        Returns:
            np.ndarray: Transposed array in NCHW format.
        """
        if len(x.shape) == 4:
            x = format_transpose(x, TensorFormat.NHWC, TensorFormat.NCHW)
        return np.asarray(x, order="C")

    def do_model2_forward_pass(self, model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """Model 2 forward pass in NCHW format"""

        x1_format = list(map(self.nhwc2nchw, x1))
        return super().do_model2_forward_pass(model2, x1_format)

    def do_model2_backward_pass(self, model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """Model 2 backward pass in NCHW format"""
        dx1_format = list(map(self.nhwc2nchw, dx1))
        return super().do_model2_backward_pass(model2, dx1_format)

    def compare_forward(
        self, model1: Model, x1: list[np.ndarray], model2: Model, x2: list[np.ndarray]
    ) -> None:
        """
        Compares forward pass outputs between two models.

        Args:
            model1: The reference model.
            x1: Input data for the reference model.
            model2: The target model to compare.
            x2: Input data for the target model.
        """
        assert len(x1) == len(x2), "x1 and x2 should have the same length"
        if verbose_test():
            logger.info("\nComparing outputs of both models...")
        for i, layer in enumerate(model1.layers, 1):
            # Skip test on layers that behave randomly
            # NOTE: Dropout uses random data and Flatten just reshape the input (it
            # make no sense to undo its work, change the format and flatten again only
            # to compare both formats)
            if not isinstance(layer, (Dropout, Flatten)):
                x1_i = self.nhwc2nchw(x1[i])
                rtol, atol = self.get_tolerance(layer)
                allclose = np.allclose(x1_i, x2[i], rtol=rtol, atol=atol)
                self.assertTrue(
                    allclose,
                    f"Forward result from layers {layer.name_with_id} differ "
                    f"({self.print_stats(x1_i, x2[i], rtol, atol)})",
                )

    def compare_backward(
        self, model1: Model, dx1: list[np.ndarray], model2: Model, dx2: list[np.ndarray]
    ) -> None:
        """
        Compares backward pass gradients between two models.

        Args:
            model1: The reference model.
            dx1: Gradient data for the reference model.
            model2: The target model to compare.
            dx2: Gradient data for the target model.
        """
        assert len(dx1) == len(dx2), "dx1 and dx2 should have the same length"
        if verbose_test():
            logger.info("\nComparing dw of both models...")
        for i, layer in reversed(list(enumerate(model2.layers, 0))):
            if isinstance(layer, (AbstractConv2D, FC)):
                rtol, atol = self.get_tolerance(layer)
                if len(layer.weights.shape) == 4:
                    # layer.dw: np.ndarray
                    transposed_dw = format_transpose(layer.dw, "OIHW", "IHWO")
                    if transposed_dw.shape == model1.layers[i].dw.shape:
                        allclose = np.allclose(
                            transposed_dw, model1.layers[i].dw, rtol=rtol, atol=atol
                        )
                        self.assertTrue(
                            allclose,
                            f"Backward dw from layer {layer.name_with_id} differ "
                            f"({self.print_stats(transposed_dw, model1.layers[i].dw, rtol, atol)})",
                        )
                elif len(layer.weights.shape) == 2:
                    # layer.dw: np.ndarray
                    transposed_dw = format_transpose(layer.dw, "OI", "IO")
                    if transposed_dw.shape == model1.layers[i].dw.shape:
                        allclose = np.allclose(
                            transposed_dw, model1.layers[i].dw, rtol=rtol, atol=atol
                        )
                        self.assertTrue(
                            allclose,
                            f"Backward dw from layer {layer.name_with_id} differ "
                            f"({self.print_stats(transposed_dw, model1.layers[i].dw, rtol, atol)})",
                        )
                else:
                    if layer.dw.shape == model1.layers[i].dw.shape:
                        allclose = np.allclose(layer.dw, model1.layers[i].dw, rtol=rtol, atol=atol)
                        self.assertTrue(
                            allclose,
                            f"Backward dw from layer {layer.name_with_id} differ "
                            f"({self.print_stats(layer.dw, model1.layers[i].dw, rtol, atol)})",
                        )
        if verbose_test():
            logger.info("\nComparing db of both models...")
        for i, layer in reversed(list(enumerate(model2.layers, 0))):
            if isinstance(layer, (AbstractConv2D, FC)) and layer.use_bias:
                rtol, atol = self.get_tolerance(layer)
                # layer.db:np.ndarray
                allclose = np.allclose(layer.db, model1.layers[i].db, rtol=rtol, atol=atol)
                self.assertTrue(
                    allclose,
                    f"Backward db from layer {layer.name_with_id} differ "
                    f"({self.print_stats(layer.db, model1.layers[i].db, rtol, atol)})",
                )
        if verbose_test():
            logger.info("\nComparing dx of both models...")
        for i, layer in reversed(list(enumerate(model2.layers, 0))):
            # Skip test on layers that behave randomly and Flatten
            if not isinstance(layer, (Dropout, Flatten)):
                rtol, atol = self.get_tolerance(layer)
                dx1_i = self.nhwc2nchw(dx1[i])
                if dx1_i.shape == dx2[i].shape:
                    allclose = np.allclose(dx1_i, dx2[i], rtol=rtol, atol=atol)
                else:
                    logger.warning(
                        f"dx shape on both models for {layer.name_with_id} differ: "
                        f"[dx1.shape: {dx1[i].shape}, dx2.shape: {dx2[i].shape}]"
                    )
                    # Try flattening both
                    allclose = np.allclose(dx1_i.flatten(), dx2[i].flatten(), rtol=rtol, atol=atol)
                self.assertTrue(
                    allclose,
                    f"Backward result from layer {layer.name_with_id} differ "
                    f"({self.print_stats(dx1_i, dx2[i], rtol, atol)})",
                )
