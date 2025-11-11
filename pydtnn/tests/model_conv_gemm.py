"""
Unitary tests for ConvGemm with different models' layers

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.CheckConvGemmModels

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckConvGemmModels

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckConvGemmModels.test_name
"""

import sys
import unittest

from pydtnn.layers.layer import LayerError
from pydtnn.model import Model

from pydtnn.tests.common import Params
from pydtnn.tests.model_common import ModelCommonTestCase
from pydtnn.utils.tensor import TensorFormat


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

    @staticmethod
    def get_model2(model_name: str, overwrite_params: dict | None = None) -> Model:
        # CPU model with convGemm
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.enable_conv_gemm = True
        params.tensor_format = TensorFormat.NHWC.upper()
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        return model2


if __name__ == "__main__":
    unittest.main()
