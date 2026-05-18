from pydtnn.metrics.abstract.metric import Metric
from pydtnn.utils import find_component


def select(name: str) -> type[Metric]:
    """
    Retrieves a metric class by its name from the metrics package.

    Args:
        name (str): The name of the metric class to retrieve.

    Returns:
        type[Metric]: The requested metric class.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)