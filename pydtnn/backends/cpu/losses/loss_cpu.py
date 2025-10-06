from abc import ABC

from pydtnn.losses import Loss


class LossCPU(Loss, ABC):
    """
    Extends a Loss class with the attributes and methods required by CPU Losses.
    """
