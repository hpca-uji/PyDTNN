import importlib, sys

def get_model(params):
    try:
        model_mod = importlib.import_module("models." + params.model)
        model = getattr(model_mod, "create_" + params.model)(params)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return model

