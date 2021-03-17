"""
Metric classes

If you want to add a new metric:
    1) create a new Python file in this directory,
    2) define your metric class as derived from Metric (or any Metric derived class),
    3) and, optionally, import your layer on this file.
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

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