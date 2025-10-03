from .relu import Relu

class LeakyRelu(Relu):
    
    def __init__(self, shape: tuple[int,...] = (1,), negative_slope: float = 0.01):
        super().__init__(shape)
        self.negative_slope:float = negative_slope

