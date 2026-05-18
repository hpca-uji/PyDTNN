from pydtnn.schedulers.abstract.scheduler import Scheduler
from pydtnn.utils import find_component


def select(name: str) -> type[Scheduler]:
    """
    Retrieve a scheduler class by its name.

    Args:
        name: The name of the scheduler component.

    Returns:
        The scheduler class type.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)