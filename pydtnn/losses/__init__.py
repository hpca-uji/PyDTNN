from pydtnn.utils import find_component
from pydtnn.losses.loss import Loss as _Loss


# TODO: remove imports and to proper dynamic import
def select(loss_func_name: str) -> type[_Loss]:
    cls = find_component("losses", loss_func_name)
    return cls
