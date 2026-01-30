import numpy as np

from pydtnn.backends.cpu.optimizers.optimizer import OptimizerCPU
from pydtnn.optimizers.nadam import Nadam
from pydtnn.backends.cpu.layers.layer import LayerCPU


class NadamCPU(Nadam[np.ndarray], OptimizerCPU):

    def initialize(self, list_layers: list[LayerCPU]) -> None:

        for layer in list_layers:
            self.context[layer.id] = dict[str, int | np.ndarray]()
            self.context[layer.id]["it"] = 0

        for w_ in layer.grad_vars.keys():
            w: np.ndarray = getattr(layer, w_)
            shape = w.shape
            momentum = np.zeros(shape, dtype=layer.model.dtype, order="C")
            velocity = np.zeros(shape, dtype=layer.model.dtype, order="C")
            self.real_memory_size += momentum.nbytes + velocity.nbytes
            self.temp_memory_size += int(2 * np.prod(shape)) * self.model.dtype.itemsize
            self.context[layer.id]["m_%s" % w_] = momentum
            self.context[layer.id]["v_%s" % w_] = velocity
            self.context[layer.id]["temp_w_%s" % w_] = vt_temp_w
            self.context[layer.id]["temp_dw_%s" % w_] = mt_temp_dw

            self.real_memory_size += self.temp_memory_size
    # ---

    def post_initialize(self) -> None:
        super().post_initialize()

        for layer_id in self.context.keys():
            for key in self.context[layer_id].keys():
                if "temp_w_" in key:
                    w_ = key.split("temp_w_")[-1]
                elif "temp_dw_" in key:
                    w_ = key.split("temp_dw_")[-1]
                else:
                    w_ = None

                if w_ is None:
                    continue
                # if w_ is not None:

                w_shape = self.context[layer_id]["m_%s" % w_].shape  # type: ignore (it is correct)
                w_shape = self.context[layer_id][key] = self.model.memory.ndarray(w_shape, dtype=self.model.dtype)
        # - end for
        self.model.memory.free(self.temp_memory_size)
    # ---

    def update(self, layer: LayerCPU) -> None:
        self.context[layer.id]["it"] += 1
        it: int = self.context[layer.id]["it"]  # type: ignore

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            w: np.ndarray
            dw: np.ndarray
        # Momentum of the weight or bias of the given layer
        m: np.ndarray = self.context[layer.id]["m_%s" % w_]  # type: ignore
        # Velocity of the weight or bias of the given layer
        v: np.ndarray = self.context[layer.id]["v_%s" % w_]  # type: ignore

        vt_temp_w: np.ndarray = self.context[layer.id]["temp_w_%s" % w_]  # type:ignore
        mt_temp_dw: np.ndarray = self.context[layer.id]["temp_dw_%s" % w_]  # type:ignore

        if not (self.are_all_zeros(w) and self.are_all_zeros(dw) or self.are_all_zeros(m) or self.are_all_zeros(v)):
            # NOTE: The operations are unrolled in order to reduce the memory consumed by intermediate copies of the variables during the operations.
            inv_beta1 = (1 - self.beta1)
            inv_beta2 = (1 - self.beta2)

            # m = self.beta1 * m + (1 - self.beta1) * dw
            np.multiply(inv_beta1, dw, dtype=self.dtype, order="C", out=mt_temp_dw)

            np.multiply(m, self.beta1, out=m,
                        dtype=self.dtype)
            np.add(m, mt_temp_dw, out=m,
                   dtype=self.dtype)

            # v = self.beta2 * v + (1 - self.beta2) * dw ** 2
            np.pow(dw, 2, dtype=self.dtype, order="C", out=mt_temp_dw)
            np.multiply(mt_temp_dw, inv_beta2, out=mt_temp_dw,
                        dtype=self.dtype)

            np.multiply(v, self.beta2, out=v,
                        dtype=self.dtype)
            np.add(v, mt_temp_dw, out=v,
                   dtype=self.dtype)

            # w -= self.learning_rate * (self.decay * w + (mt / np.sqrt(vt + epsilon))) ==>
            # w -= (self.learning_rate * self.decay * w) + (self.learning_rate * (mt / np.sqrt(vt + epsilon))))

            # w -= (self.learning_rate * self.decay * w)
            np.multiply((self.learning_rate * self.decay), w, dtype=self.dtype, order="C", out=mt_temp_dw)

            np.subtract(w, mt_temp_dw, out=w,
                        dtype=self.dtype)

            # (
            # mt = (m + (1 - self.beta1) * dw) / (1 - self.beta1 ** it)
            np.multiply(inv_beta1, dw, dtype=self.dtype, order="C", out=mt_temp_dw)
            np.divide(mt_temp_dw, (inv_beta1 ** it), out=mt_temp_dw,
                      dtype=self.dtype)
            np.add(m, mt_temp_dw, out=mt_temp_dw,
                   dtype=self.dtype)
            # )

            # (
            # vt = v / (1 - self.beta2 ** it)
            np.divide(v, (inv_beta2 ** it), dtype=self.dtype, order="C", out=vt_temp_w)
            # )

            # w -= (self.learning_rate * (mt / np.sqrt(vt + epsilon))))
            np.add(vt_temp_w, self.epsilon, out=vt_temp_w,
                   dtype=self.dtype)
            np.sqrt(vt_temp_w, out=vt_temp_w,
                    dtype=self.dtype)
            np.divide(mt_temp_dw, vt_temp_w, out=mt_temp_dw,
                      dtype=self.dtype)
            np.multiply(self.learning_rate, mt_temp_dw, out=mt_temp_dw,
                        dtype=self.dtype, order="C")
            np.subtract(w, mt_temp_dw, out=w,
                        dtype=self.dtype)
        # else: continue
