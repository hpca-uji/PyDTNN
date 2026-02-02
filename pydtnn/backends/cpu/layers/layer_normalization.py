import numpy as np

from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.backends.cpu.layers.layer import LayerCPU


class LayerNormalizationCPU(LayerNormalization[np.ndarray], LayerCPU):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Check how to initialize this parameters outside (in the initalization layer)
        mu = np.mean(x, axis=self.axis, keepdims=True)
        xc = (x - mu)
        var = np.mean(xc ** 2, axis=self.axis, keepdims=True)

        # self.std = np.sqrt(var + self.epsilon)
        self.std = np.add(var, self.epsilon)
        np.sqrt(self.std, out=self.std, dtype=self.model.dtype)

        # self.xn = xc / self.std
        self.xn = np.divide(xc, self.std, dtype=self.model.dtype)

        # y = self.gamma * self.xn + self.beta
        y = np.multiply(self.gamma, self.xn, dtype=self.model.dtype)
        np.add(y, self.beta, out=y)

        return y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        # if self.need_dx:
        # dy = dy * self.gamma
        np.mutliply(dy, self.gamma, out=dy)

        # dx = dy - self.xn * np.mean(dy * self.xn, self.axis, keepdims=True)
        dx = np.mean(dy * self.xn, self.axis, keepdims=True)
        np.multiply(self.xn, dx, out=dx, dtype=self.model.dtype)
        np.subtract(dy, dx, out=dx)

        # dx -= np.mean(dy, self.axis, keepdims=True)
        _mean = np.mean(dy, self.axis, keepdims=True)
        np.subtract(dx, _mean, out=dx)

        # dx /= self.std
        np.divide(dx, self.std, out=dx, dtype=self.model.dtype)
        return dx
