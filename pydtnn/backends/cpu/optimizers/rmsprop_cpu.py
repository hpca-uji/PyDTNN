import numpy as np

from pydtnn.backends.cpu.optimizers import OptimizerCPU
from pydtnn.optimizers import RMSProp

from pydtnn.backends.cpu.layers import LayerCPU

class RMSPropCPU(OptimizerCPU, RMSProp):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer] = dict[str, np.ndarray]()
                for w_ in list_grad_vars:
                    w:np.ndarray = getattr(layer, w_)
                    self.context[layer]["cache_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype)

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache:np.ndarray = self.context[layer]["cache_%s" % w_]
            w:np.ndarray
            dw:np.ndarray
            
            if not (self.are_all_zeros(w) and self.are_all_zeros(dw) and self.are_all_zeros(cache)):
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.

                #cache = self.rho * cache + (1 - self.rho) * dw ** 2
                cache *= self.rho
                _dw = dw ** 2
                _dw *= (1 - self.rho)
                cache += _dw
                #w -= self.learning_rate * (self.decay * w + (dw / np.sqrt(cache + self.epsilon)))
                w -= (self.learning_rate * self.decay) * w
                _cache = cache + self.epsilon
                np.sqrt(_cache, out=_cache)
                _dw = dw / _cache 
                _dw *= self.learning_rate
                w -= _dw
            #else: continue
