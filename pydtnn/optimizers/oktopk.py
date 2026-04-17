import logging
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
from warnings import warn

from pydtnn.abstract.layerable import Layerable
from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

if TYPE_CHECKING:
    from pydtnn.model import Model


class OkTopk[T: Array](Optimizer[T]):
    """
    SGD Ok-Topk Optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9,
                 tau: int = 64, tau_prime: int = 32, density: float = 0.01, min_k_layer: int = 10):

        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.residuals = {}
        self.tau = tau
        self.tau_prime = tau_prime
        self.density = density
        self.min_k_layer = min_k_layer
        self.iterations = {}
        self.all_local_th = {}
        self.all_global_th = {}
        self.all_residuals = {}
        self.all_boundaries = {}
        self.info_messages = set()

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["momentum"] = self.momentum
        props["tau"] = self.tau
        props["tau_prime"] = self.tau_prime
        props["density"] = self.density
        props["min-k-layer"] = self.min_k_layer

        return props

    def _model_init(self, list_layers: list[Layerable]) -> None:
        super()._model_init(list_layers)

        if self.model.model_sync_freq >= 0:
            warn("Optimizer does model sync but global model sync is also enabled!", RuntimeWarning)

        if not self.model.shared_storage:
            raise NotImplementedError("OkTopK optimizer does not support Federated Learing (unbalanced datasets)!")

    @classmethod
    def from_model(cls, model: "Model") -> "OkTopk":
        return OkTopk(learning_rate=model.learning_rate,
                      momentum=model.optimizer_momentum,
                      tau=model.optimizer_tau,
                      tau_prime=model.optimizer_tau_prime,
                      density=model.optimizer_density)
