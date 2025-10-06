"""
Metric CPU classes

If you want to add a new CPU metric:
    1) create a new Python file in this directory,
    2) define your CPU metric class as derived from MetricCPU and, optionally, other Metric derived class,
    3) and, optionally, import your metric on this file.
"""

from pydtnn.backends.cpu.metrics.metric_cpu import MetricCPU
from pydtnn.utils import get_derived_classes

# Search this module for MetricGPU derived classes and expose them
get_derived_classes(MetricCPU, locals())
