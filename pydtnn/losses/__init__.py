from pydtnn.losses.abstract.loss import Loss
from pydtnn.utils import find_component


def select(name: str) -> type[Loss]:
    """
    Selects a loss class by its name.

    Args:
        name (str): The name of the loss class to retrieve.

    Returns:
        type[Loss]: The requested loss class.

    Raises:
        AssertionError: If the package is not defined.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)