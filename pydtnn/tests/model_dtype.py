import unittest

import numpy as np

from pydtnn.layers.layer import LayerError
from pydtnn.model import Model
from pydtnn.tests.common import Params
from pydtnn.tests.model_common import ModelCommonTestCase


class ModelDTypeTestCase(ModelCommonTestCase):
    """
    Tests that two models with different parameters lead to the same results
    """
    # NOTE: Delete parent test to prevent re-export and re-testing
    global ModelCommonTestCase
    del ModelCommonTestCase

    # Compares results between an XX model {self.model1_desc} and other {self.model1_desc}
    model1_desc = "using float32"
    model2_desc = "using float64"

    @staticmethod
    def get_model2(model_name: str, overwrite_params: dict | None = None) -> Model:
        # CPU model with float64
        params = Params()
        # Begin of params configuration
        params.model_name = model_name
        params.dtype = np.float64
        # End of params configuration
        params_dict = vars(params)
        if overwrite_params is not None:
            params_dict.update(overwrite_params)
        try:
            model2 = Model(**params_dict)
        except LayerError as exc:
            raise unittest.SkipTest(f"Model {model_name} incompatible with {params_dict['dataset_name']}") from exc
        return model2

    @staticmethod
    def do_model2_forward_pass(model2: Model, x1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 forward pass
        """
        from pydtnn.tests.model_common import ModelCommonTestCase
        return [x.astype(np.float32) for x in ModelCommonTestCase.do_model2_forward_pass(model2, [x.astype(np.float64) for x in x1])]

    @staticmethod
    def do_model2_backward_pass(model2: Model, dx1: list[np.ndarray]) -> list[np.ndarray]:
        """
        Model 2 backward pass
        """
        from pydtnn.tests.model_common import ModelCommonTestCase
        return [dx.astype(np.float32) for dx in ModelCommonTestCase.do_model2_backward_pass(model2, [dx.astype(np.float64) for dx in dx1])]