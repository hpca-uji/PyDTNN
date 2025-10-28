from pydtnn.losses.loss import Loss
import numpy as np

class LossCPU(Loss[np.ndarray]):
    """
    Extends a Loss class with the attributes and methods required by CPU Losses.
    """
