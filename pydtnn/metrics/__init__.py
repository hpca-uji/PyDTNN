from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy as _CategoricalAccuracy
from pydtnn.metrics.categorical_hinge import CategoricalHinge as _CategoricalHinge
from pydtnn.metrics.categorical_mae import CategoricalMAE as _CategoricalMAE
from pydtnn.metrics.categorical_mse import CategoricalMSE as _CategoricalMSE
from pydtnn.metrics.metric import Metric as _Metric
from pydtnn.metrics.regression_mae import RegressionMAE as _RegressionMAE
from pydtnn.metrics.regression_mse import RegressionMSE as _RegressionMSE
from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix as _BinaryConfusionMatrix
from pydtnn.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrix as _MulticlassConfusionMatrix

metric_format = {"categorical_accuracy": "acc: %5.2f%%",
                 "categorical_cross_entropy": "cce: %.7f",
                 "binary_cross_entropy": "bce: %.7f",
                 "categorical_hinge": "hin: %.7f",
                 "categorical_mse": "mse: %.7f",
                 "categorical_mae": "mae: %.7f",
                 "regression_mse": "mse: %.7f",
                 "regression_mae": "mae: %.7f"}


# TODO: remove imports and to proper dynamic import
def select(loss_func_name: str) -> type[_Metric]:
    # From snake to camel, if it's necessary
    _loss_func_name = loss_func_name.split("_")
    if len(_loss_func_name) > 1:
        _loss_func_name = "".join(map(lambda x: x.lower().capitalize(), _loss_func_name))
    else:
        _loss_func_name = loss_func_name

    match _loss_func_name:
        case _CategoricalAccuracy.__name__:
            return _CategoricalAccuracy
        case _CategoricalHinge.__name__:
            return _CategoricalHinge
        case _CategoricalMAE.__name__:
            return _CategoricalMAE
        case _CategoricalMSE.__name__:
            return _CategoricalMSE
        case _RegressionMAE.__name__:
            return _RegressionMAE
        case _RegressionMSE.__name__:
            return _RegressionMSE
        case _BinaryConfusionMatrix.__name__:
            return _BinaryConfusionMatrix
        case _MulticlassConfusionMatrix.__name__:
            return _MulticlassConfusionMatrix
        case _:
            raise NotImplementedError(f"\'{loss_func_name}\' is not implemented!")
