from pydtnn.backends.numpy.activations.activation import ActivationNumpy


class ActivationCupy(ActivationNumpy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"
