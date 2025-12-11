import numpy as np

from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.backends.cpu.layers.layer import LayerCPU


class LayerNormalizationCPU(LayerNormalization[np.ndarray], LayerCPU):
    def forward(self, x):
        mu = np.mean(x, axis=self.axis, keepdims=True)
        xc = (x - mu)
        var = np.mean(xc ** 2, axis=self.axis, keepdims=True)

        self.std = np.sqrt(var + self.epsilon)
        self.xn = xc / self.std
        y = self.gamma * self.xn + self.beta

        return y

    def backward(self, dy):
        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)
        if self.need_dx:
            dy = dy * self.gamma
            dx = dy - self.xn * np.mean(dy * self.xn, self.axis, keepdims=True)
            dx -= np.mean(dy, self.axis, keepdims=True)
            dx /= self.std
            return dx
