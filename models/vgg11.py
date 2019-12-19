from NN_model import *
from NN_layer import *

def create_vgg11(comm):
    model = Model(comm)
    model.add( Input(shape=(224, 224, 3)) )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3, 3), padding=1, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(2,2), func='max', stride=1) )
    model.add( Conv2D(nfilters=128, filter_shape=(3, 3, 64), padding=1, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(2,2), func='max', stride=1) )
    model.add( Conv2D(nfilters=256, filter_shape=(3, 3, 128), padding=1, stride=1, activation="relu") )
    model.add( Conv2D(nfilters=256, filter_shape=(3, 3, 256), padding=1, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(2,2), func='max', stride=1) )
    model.add( Conv2D(nfilters=512, filter_shape=(3, 3, 256), padding=1, stride=1, activation="relu") )
    model.add( Conv2D(nfilters=512, filter_shape=(3, 3, 512), padding=1, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(2,2), func='max', stride=1) )
    model.add( Conv2D(nfilters=512, filter_shape=(3, 3, 512), padding=1, stride=1, activation="relu") )
    model.add( Conv2D(nfilters=512, filter_shape=(3, 3, 512), padding=1, stride=1, activation="relu") )
    model.add( Pool2D(pool_shape=(2,2), func='max', stride=1) )
    model.add( Flatten() )
    model.add( FC(shape=(4096,), activation="relu") )
    model.add( FC(shape=(4096,), activation="relu") )
    model.add( FC(shape=(1000,), activation="sigmoid") )
    return model
