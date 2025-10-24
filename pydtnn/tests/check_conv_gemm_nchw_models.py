"""
Unitary tests for ConvGemm with different models' layers

For running all the tests quietly, execute the next command:
    python -um unittest pydtnn.tests.CheckConvGemmNCHWModels

For running all the tests verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckConvGemmNCHWModels

For running an individual test verbosely, execute the next command:
    python -um unittest -v pydtnn.tests.CheckConvGemmNCHWModels.test_name
"""

import sys
import unittest

from pydtnn.model import Model
from pydtnn.tests import CheckConvGemmModels
from pydtnn.losses.loss import Loss
from pydtnn.utils.tensor import TensorFormat


class CheckConvGemmNCHWModels(CheckConvGemmModels):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using Im2Col+MM"
    model2_desc = "using ConvGemm"

    @staticmethod
    def get_model1_and_loss_func(model_name: str) -> tuple[Model, Loss]:
        model1, loss_func = CheckConvGemmModels.get_model1_and_loss_func(model_name,
                                                                         overwrite_params={'tensor_format': TensorFormat.NCHW.upper()})
        return model1, loss_func

    @staticmethod
    def get_model2(model_name: str) -> Model:
        model2 = CheckConvGemmModels.get_model2(model_name,
                                                overwrite_params={'tensor_format': TensorFormat.NCHW.upper()})
        return model2


if __name__ == '__main__':
    try:
        Model()
    except NameError:
        sys.exit(-1)
    unittest.main()
