"""
Tests for verifying model behavior and consistency across different data types.
"""

import logging
import unittest

import numpy as np

from pydtnn.layers.abstract.layer import LayerError
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.tests.abstract.common import Params
from pydtnn.tests.abstract.model_common import ModelCommonTestCase

__all__ = ("ModelDTypeTestCase",)

logger = logging.getLogger(__name__)


class ModelDTypeTestCase(ModelCommonTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """

    # NOTE: Delete parent test to prevent re-export and re-testing
    global ModelCommonTestCase
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model2_desc}
    model1_desc = "using float32"
    model2_desc = "using float64"

    rtol_default = 1e-4
    atol_default = 1e-5
    rtol_dict = {
        AdditionBlock: 5e-3,
        ConcatenationBlock: 1e-1,
        BatchNormalization: 1e-5,
        Conv2D: 1e-4,
    }
    atol_dict = {
        AdditionBlock: 5e-3,
        ConcatenationBlock: 1e-1,
        Conv2D: 1e-5,
        BatchNormalization: 1e-4,
    }

    def get_model2(self, model_name: str, overwrite_params: dict | None = None) -> Model:
        """
        Constructs and returns a model instance configured with float64 precision.

        Args:
            model_name: The name of the model to instantiate.
            overwrite_params: Optional dictionary of parameters to override defaults.

        Returns:
            A Model instance configured for float64.

        Raises:
            unittest.SkipTest: If the model is incompatible with the dataset.
        """
        # CPU model with float64
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.dtype = np.dtype(np.float64)
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(
                f"Model {model_name} incompatible with {params_dict['dataset_name']}"
            ) from exc
        model2._model_init()
        return model2
