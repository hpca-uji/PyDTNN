from NN_model import *
from NN_layer import *

def create_resnet50(comm, tracing):
    model = Model(comm, tracing)
    model.add( Input(shape=(229, 229, 3)) )
    return model
