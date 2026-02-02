from pydtnn.libs import numpy as np

from pydtnn.backends.cpu.optimizers.optimizer import OptimizerCPU
from pydtnn.optimizers.sgd import SGD

from pydtnn.backends.cpu.layers.layer import LayerCPU


class SGDCPU(SGD[np.ndarray], OptimizerCPU):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        temp_memory_size = []

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())
            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, np.ndarray]()  # type: ignore
                for w_ in list_grad_vars:
                    w: np.ndarray = getattr(layer, w_)
                    velocity: np.ndarray = np.zeros(w.shape, dtype=layer.model.dtype)
                    temp_w = None
                    temp_v = None
                    self.real_memory_size += velocity.nbytes

                    temp_memory_size.append(int(2 * np.prod(w.shape)) * self.model.dtype.itemsize)
                    self.context[layer.id]["velocity_%s" % w_] = velocity
                    self.context[layer.id]["temp_w_%s" % w_] = temp_w
                    self.context[layer.id]["temp_v_%s" % w_] = temp_v

        self.temp_memory_size += self.model.memory_cls._total(*temp_memory_size)
        self.real_memory_size += self.temp_memory_size
    # ---

    def post_initialize(self) -> None:
        super().post_initialize()
        for layer_id in self.context.keys():
            with self.model.memory:
                for key in self.context[layer_id].keys():
                    if "temp_w_" in key:
                        w_ = key.split("temp_w_")[-1]
                    elif "temp_v_" in key:
                        w_ = key.split("temp_v_")[-1]
                    else:
                        w_ = None

                    if w_ is None:
                        continue
                    # if w_ is not None:

                    w_shape = self.context[layer_id]["velocity_%s" % w_].shape  # type: ignore (it is correct)
                    w_shape = self.context[layer_id][key] = self.model.memory.ndarray(w_shape, dtype=self.model.dtype)
        # - end for
    # ---

    def update(self, layer: LayerCPU) -> None:
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity: np.ndarray = self.context[layer.id]["velocity_%s" % w_]  # type: ignore
            temp_w: np.ndarray = self.context[layer.id]["temp_w_%s" % w_]  # type: ignore
            temp_v: np.ndarray = self.context[layer.id]["temp_v_%s" % w_]  # type: ignore
            w: np.ndarray
            dw: np.ndarray

            if (w is not None and dw is not None) and not (self.are_all_zeros(velocity) and self.are_all_zeros(w) and self.are_all_zeros(dw)):
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
                    np.multiply(velocity, self.momentum, dtype=self.dtype, out=temp_v)
                    np.add(temp_v, dw, out=temp_v,
                           dtype=self.dtype)
                else:
                    # np.copyto(temp_v, velocity)
                    temp_v[:] = velocity
                np.multiply(w, self.decay, dtype=self.dtype, out=temp_w)
                np.add(temp_w, temp_v, out=temp_w,
                       dtype=self.dtype)
                np.multiply(temp_w, self.learning_rate, out=temp_w,
                            dtype=self.dtype)
                np.subtract(w, temp_w, out=w,
                            dtype=self.dtype)
            # else: continue
