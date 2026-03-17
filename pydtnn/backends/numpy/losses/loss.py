import logging
logger = logging.getLogger(__name__)

from pydtnn.losses.loss import Loss
from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class LossNumpy(Loss[np.ndarray]):
    """
    Extends a Loss class with the attributes and methods required by CPU Losses.
    """

    def _model_init(self) -> None:
        super()._model_init()
        self.dx = np.ndarray(self.shape, dtype=self.model.dtype)
        self.memory_used += self.dx.nbytes
