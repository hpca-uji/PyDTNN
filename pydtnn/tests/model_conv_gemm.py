"""
Tests for verifying the equivalence of convolution implementations using Im2Col+MM versus ConvGemm.
"""

import logging
import unittest

from pydtnn.layers.abstract.layer import LayerError
from pydtnn.libs.convGemm import is_conv_gemm_available
from pydtnn.model import Model
from pydtnn.tests.abstract.common import Params
from pydtnn.tests.abstract.model_common import ModelCommonTestCase
from pydtnn.utils.tensor import TensorFormat

__all__ = ("ModelConvGemmTestCase",)

logger = logging.getLogger(__name__)


@unittest.skipUnless(is_conv_gemm_available, "requires ConvGemm")
class ModelConvGemmTestCase(ModelCommonTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global ModelCommonTestCase
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using Im2Col+MM"
    model2_desc = "using ConvGemm"

    def get_model2(self, model_name: str, overwrite_params: dict | None = None) -> Model:
        """
        Constructs and returns a model configured to use the ConvGemm backend.

        Args:
            model_name: The name of the model to instantiate.
            overwrite_params: Optional dictionary of parameters to override defaults.

        Returns:
            A configured Model instance.

        Raises:
            unittest.SkipTest: If the model is incompatible with the dataset.
        """
        # CPU model with convGemm
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.backend = "cpu;conv_2d:gemm"
        params.tensor_format = TensorFormat.NHWC.upper()
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        model2._model_init()
        return model2

    @unittest.skip("FIXME: Test error (disabled)")
    def test_resnet34(self):
        """Compares results between a Densenet model using A and other using B."""
        super().test_resnet34()

    @unittest.skip("FIXME: Test error (disabled)")
    def test_densenet(self):
        """Compares results between a Densenet model using CPU and other using GEMM."""
        super().test_densenet()
