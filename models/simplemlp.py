from NN_model import *
from NN_layer import *

def create_simplemlp(comm, tracing):
    model = Model(comm, tracing)
    model.add( Input(shape=(28, 28, 1)) )
    model.add( Flatten() )#
    model.add( FC(shape=(512,), activation="sigmoid") )
    model.add( FC(shape=(512,), activation="sigmoid") )
    model.add( FC(shape=(512,), activation="sigmoid") )
    model.add( FC(shape=(10,), activation="sigmoid") )
    return model
