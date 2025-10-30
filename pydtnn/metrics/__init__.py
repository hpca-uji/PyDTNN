from pydtnn.metrics.metric import Metric as _Metric


# TODO: remove imports and to proper dynamic import
def select(metric_func_name: str) -> type[_Metric]:
    from pydtnn.metrics.categorical_accuracy import CategoricalAccuracy
    from pydtnn.metrics.categorical_hinge import CategoricalHinge
    from pydtnn.metrics.categorical_mae import CategoricalMAE
    from pydtnn.metrics.categorical_mse import CategoricalMSE
    from pydtnn.metrics.regression_mae import RegressionMAE
    from pydtnn.metrics.regression_mse import RegressionMSE
    from pydtnn.metrics.binary_confusion_matrix import BinaryConfusionMatrix
    from pydtnn.metrics.multiclass_confusion_matrix import MulticlassConfusionMatrix

    metric = {
        "categorical_accuracy": CategoricalAccuracy,
        "categorical_hinge": CategoricalHinge,
        "categorical_mae": CategoricalMAE,
        "categorical_mse": CategoricalMSE,
        "regression_mae": RegressionMAE,
        "regression_mse": RegressionMSE,
        "binary_confusion_matrix": BinaryConfusionMatrix,
        "multiclass_confusion_matrix": MulticlassConfusionMatrix,
    }

    try:
        cls = metric[metric_func_name]
    except KeyError:
        raise ValueError(f"Metric {metric_func_name!r} not found!") from None

    return cls
