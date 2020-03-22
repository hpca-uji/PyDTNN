from NN_model import *
from NN_layer import *
from NN_activation import *

def create_inception_v3(params):
    model = Model(params, comm=params.comm, 
                      blocking_mpi=params.blocking_mpi,
                      tracing=params.tracing, 
                      dtype=params.dtype)
    model.add( Input(shape=(225, 225, 3)) )
    model.add( Conv2D(nfilters=32, filter_shape=(3, 3), padding=0, stride=2, activation=Relu()) )
    model.add( Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) )     #¿?¿?¿?¿?¿?
    model.add( Conv2D(nfilters=80, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192,filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) )     #¿?¿?¿?¿?¿?
    model.add( Conv2D(nfilters=64, filter_shape=(1, 2), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=48, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(5, 5), padding=2, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=32, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=48, filter_shape=(5, 5), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=2, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(3, 3), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=48, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(5, 5), padding=2, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384,filter_shape=(3, 3), padding=0, stride=2, activation=Relu()) )      #¿?¿?¿?¿?¿? 
    model.add( Conv2D(nfilters=64, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=96, filter_shape=(3, 3), padding=0, stride=2, activation=Relu()) )     #¿?¿?¿?¿?¿?
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) ) 
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=128, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=128, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=128, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=128, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=128, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=128, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=160, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=160, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=160, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=160, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=160, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=160, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=320, filter_shape=(3, 3), padding=1, stride=2, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(1, 7), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(7, 1), padding=3, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=192, filter_shape=(3, 3), padding=0, stride=2, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=2) )     #PADDING=1
    model.add( Conv2D(nfilters=320, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(1, 3), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 1), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=448, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(1, 3), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 1), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='avg', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=320, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(1, 3), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 1), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=448, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(1, 3), padding=0, stride=1, activation=Relu()) )
    model.add( Conv2D(nfilters=384, filter_shape=(3, 1), padding=1, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(3,3), func='max', stride=1) )     #PADDING=1
    model.add( Conv2D(nfilters=192, filter_shape=(1, 1), padding=0, stride=1, activation=Relu()) )
    model.add( Pool2D(pool_shape=(8,8), func='avg', stride=1) )     #PADDING!!
    #model.add( Flatten() )
    model.add( FC(shape=(1000,), activation=Softmax()) )
    return model
