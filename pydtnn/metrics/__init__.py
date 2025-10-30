from pydtnn.metrics.metric import Metric as _Metric
from pydtnn.utils import find_component


def select(metric_func_name: str) -> type[_Metric]:
    cls = find_component("metrics", metric_func_name)
    return cls
