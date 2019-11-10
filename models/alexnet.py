from NN_model import *
from NN_layer import *

def create_alexnet(comm):
    model = Model(comm)
    model.add( Input(shape=(227, 227, 3)) )
    model.add( Conv2D(nfilters=64, filter_shape=(11, 11, 3), padding=0, stride=4, activation="relu") )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) )
    model.add( Conv2D(nfilters=192, filter_shape=(5, 5, 64), padding=2, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 3, 192), padding=1, stride=1, activation="relu") )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 3, 384), padding=1, stride=1, activation="relu") )
    model.add( Conv2D(nfilters=256, filter_shape=(3, 3, 384), padding=1, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) )
    model.add( Flatten() )
    model.add( FC(shape=(4096,), activation="relu") )
    model.add( FC(shape=(4096,), activation="relu") )
    model.add( FC(shape=(1000,), activation="sigmoid") )
    return model

