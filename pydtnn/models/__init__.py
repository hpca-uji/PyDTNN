import importlib, sys
from model import *


def get_model(params):
    try:
        model_mod = importlib.import_module("models." + params.model)
        model = Model(params, comm=params.comm,
                      non_blocking_mpi=params.non_blocking_mpi,
                      enable_gpu=params.enable_gpu,
                      enable_gpudirect=params.enable_gpudirect,
                      enable_nccl=params.enable_nccl,
                      dtype=params.dtype,
                      tracing=params.tracing,
                      tracer_output=params.tracer_output
                      )
        model = getattr(model_mod, "create_" + params.model)(model)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return model
