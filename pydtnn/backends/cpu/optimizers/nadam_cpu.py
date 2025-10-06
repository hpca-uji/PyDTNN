import numpy as np

from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import Nadam
from pydtnn.backends.cpu.layers import LayerCPU


class NadamCPU(OptimizerCPU, Nadam):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            self.context[layer] = dict[str, int | np.ndarray]()
            self.context[layer]["it"] = 0

            for w_ in layer.grad_vars.keys():
                w: np.ndarray = getattr(layer, w_)
                self.context[layer]["m_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)
                self.context[layer]["v_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        self.context[layer]["it"] += 1
        it: int = self.context[layer]["it"]

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            w: np.ndarray
            dw: np.ndarray
            # Momentum of the weight or bias of the given layer
            m: np.ndarray = self.context[layer]["m_%s" % w_]
            # Velocity of the weight or bias of the given layer
            v: np.ndarray = self.context[layer]["v_%s" % w_]

            if not (self.are_all_zeros(w) and self.are_all_zeros(dw) or self.are_all_zeros(m) or self.are_all_zeros(v)):

                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.
                # m = self.beta1 * m + (1 - self.beta1) * dw
                m *= self.beta1
                _dw: np.ndarray = (1 - self.beta1) * dw
                m += _dw
                # v = self.beta2 * v + (1 - self.beta2) * dw ** 2
                v *= self.beta2
                _dw = dw ** 2
                _dw *= (1 - self.beta2)
                v += _dw

                # mt = (m + (1 - self.beta1) * dw) / (1 - self.beta1 ** it)
                mt = (1 - self.beta1) * dw
                mt /= (1 - self.beta1 ** it)
                mt += m

                # vt = v / (1 - self.beta2 ** it)
                vt = v / (1 - self.beta2 ** it)

                # w -= self.learning_rate * (self.decay * w + (mt / np.sqrt(vt + epsilon)))
                w -= (self.learning_rate * self.decay) * w
                vt += self.epsilon
                np.sqrt(vt, out=vt)
                mt /= vt
                mt *= self.learning_rate
                w -= mt
            # else: continue
