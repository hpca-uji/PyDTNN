"""
Metric classes

If you want to add a new metric:
    1) create a new Python file in this directory,
    2) define your metric class as derived from Metric (or any Metric derived class),
    3) and, optionally, import your layer on this file.
"""

from .categorical_accuracy import CategoricalAccuracy
from .categorical_hinge import CategoricalHinge
from .categorical_mae import CategoricalMAE
from .categorical_mse import CategoricalMSE
from .metric import Metric
from .regression_mae import RegressionMAE
from .regression_mse import RegressionMSE
from ..utils import get_derived_classes

# Aliases
categorical_accuracy = CategoricalAccuracy
categorical_hinge = CategoricalHinge
categorical_mse = CategoricalMSE
categorical_mae = CategoricalMAE
regression_mse = RegressionMSE
regression_mae = RegressionMAE

# Search this module for Metric derived classes and expose them
get_derived_classes(Metric, locals())

metric_format = {"categorical_accuracy": "acc: %5.2f%%",
                 "categorical_cross_entropy": "cce: %.7f",
                 "binary_cross_entropy": "bce: %.7f",
                 "categorical_hinge": "hin: %.7f",
                 "categorical_mse": "mse: %.7f",
                 "categorical_mae": "mae: %.7f",
                 "regression_mse": "mse: %.7f",
                 "regression_mae": "mae: %.7f"}
