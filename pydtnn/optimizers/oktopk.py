from typing import TYPE_CHECKING

import numpy as np

from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.utils.constants import Array

if TYPE_CHECKING:
    from pydtnn.model import Model


class OkTopk[T: Array](Optimizer[T]):
    """
    SGD Ok-Topk Optimizer
    """

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9, dtype: np.dtype = np.float32,
                 nprocs: int = 1, comm=None, rank: int = 0, tau: int = 64, tau_prime: int = 32, density: float = 0.01, min_k_layer: int = 10):

        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nprocs = nprocs
        self.residuals = {}
        self.dtype = dtype
        self.comm = comm
        self.rank = rank
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

    @classmethod
    def from_model(cls, model: "Model") -> "OkTopk":
        return OkTopk(learning_rate=model.learning_rate,
                      momentum=model.optimizer_momentum,
                      nprocs=model.nprocs,
                      dtype=model.dtype,
                      comm=model.comm,
                      rank=model.rank,
                      tau=model.optimizer_tau,
                      tau_prime=model.optimizer_tau_prime,
                      density=model.optimizer_density)
