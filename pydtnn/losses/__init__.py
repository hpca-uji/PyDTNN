from pydtnn.losses.loss import Loss as _Loss


# TODO: remove imports and to proper dynamic import
def select(loss_func_name: str) -> type[_Loss]:
    from pydtnn.losses.binary_cross_entropy import BinaryCrossEntropy
    from pydtnn.losses.categorical_cross_entropy import CategoricalCrossEntropy

    loss = {
        "binary_cross_entropy": BinaryCrossEntropy,
        "categorical_cross_entropy": CategoricalCrossEntropy
    }

    try:
        cls = loss[loss_func_name]
    except KeyError:
        raise ValueError(f"Loss {loss_func_name!r} not found!") from None

    return cls
