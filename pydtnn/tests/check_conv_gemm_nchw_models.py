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


class Params:
    pass


class CheckConvGemmNCHWModels(CheckConvGemmModels):
    """
    Tests that two models with different parameters lead to the same results
    """

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using Im2Col+MM"
    model2_desc = "using ConvGemm"

    @staticmethod
    def get_model1_and_loss_func(model_name):
        model1, loss_func = CheckConvGemmModels.get_model1_and_loss_func(model_name,
                                                                         overwrite_params={'tensor_format': 'NCHW'})
        return model1, loss_func

    @staticmethod
    def get_model2(model_name):
        model2 = CheckConvGemmModels.get_model2(model_name,
                                                overwrite_params={'tensor_format': 'NCHW'})
        return model2


if __name__ == '__main__':
    try:
        Model()
    except NameError:
        sys.exit(-1)
    unittest.main()
