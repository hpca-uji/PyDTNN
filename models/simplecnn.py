from NN_model import *
from NN_layer import *

def create_simplecnn(comm):
    model = Model(comm)
    model.add( Input(shape=(28, 28, 1)) )
    model.add( Conv2D(nfilters=4, filter_shape=(3, 3, 1), padding=0, stride=1, activation="sigmoid") )
    model.add( Pool2D(pool_shape=(2,2), func='max') )
    model.add( Conv2D(nfilters=6, filter_shape=(3, 3, 4), padding=0, stride=1, activation="sigmoid") )
    model.add( Pool2D(pool_shape=(2,2), func='max') )
    model.add( Flatten() )#
    #model.add( FC(shape=(128,), activation="sigmoid") )
    #model.add( Dropout(prob=0.5) )
    #model.add( FC(shape=(36,), activation="sigmoid") )
    model.add( FC(shape=(10,), activation="sigmoid") )
    return model
