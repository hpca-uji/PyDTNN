import importlib, sys
from NN_model import *

def get_model(params):
    try:
        model_mod = importlib.import_module("models." + params.model)
        model = Model(params, comm=params.comm, 
                              non_blocking_mpi=params.non_blocking_mpi,
                              tracing=params.tracing,
                              enable_gpu=params.enable_gpu, 
                              dtype=params.dtype)
        model = getattr(model_mod, "create_" + params.model)(model)
       
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return model

