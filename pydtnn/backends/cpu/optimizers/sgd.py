import numpy as np

from pydtnn.backends.cpu.optimizers.optimizer import OptimizerCPU
from pydtnn.optimizers.sgd import SGD

from pydtnn.backends.cpu.layers.layer import LayerCPU


class SGDCPU(SGD[np.ndarray], OptimizerCPU):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())
            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, np.ndarray]()  # type: ignore 
                for w_ in list_grad_vars:
                    w: np.ndarray = getattr(layer, w_)
                    self.context[layer.id]["velocity_%s" % w_] = np.zeros_like(w, dtype=layer.model.dtype, order="C")

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity: np.ndarray = self.context[layer.id]["velocity_%s" % w_]  # type: ignore 
            w: np.ndarray
            dw: np.ndarray

            if not (self.are_all_zeros(velocity) and self.are_all_zeros(w) and self.are_all_zeros(dw)):
                # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.

                # velocity = self.momentum * velocity + dw
                
                np.multiply(velocity, self.momentum, out=velocity, 
                            dtype=self.dtype)
                np.add(velocity, dw, out=velocity, 
                       dtype=self.dtype)

                # if self.nesterov:
                #    w -= self.learning_rate * (self.decay * w + dw + self.momentum * velocity)
                # else:
                #    w -= self.learning_rate * (self.decay * w + velocity)
                if self.nesterov:
                    v = np.multiply(velocity, self.momentum, dtype=self.dtype, order="C")
                    np.add(v, dw, out=v, 
                           dtype=self.dtype)
                else:
                    v = velocity
                temp_w = np.multiply(w, self.decay, dtype=self.dtype, order="C")
                np.add(temp_w, v, out=temp_w,
                       dtype=self.dtype)
                np.multiply(temp_w, self.learning_rate, out=temp_w, 
                            dtype=self.dtype)
                np.subtract(w, temp_w, out=w,
                            dtype=self.dtype)
                del temp_w
            # else: continue
