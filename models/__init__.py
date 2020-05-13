import importlib, sys, numpy as np

def get_dataset(params):
    try:
        dataset_name = {"mnist": "MNIST", "cifar10": "CIFAR10", "imagenet": "ImageNet"}
        dtype = getattr(np, params.dtype)
        dataset_mod = importlib.import_module("datasets.NN_dataset")
        dataset_obj = getattr(dataset_mod, dataset_name[params.dataset])
        dataset = dataset_obj(train_path         = params.dataset_train_path, 
                              test_path          = params.dataset_test_path,
                              model              = params.model,
                              test_as_validation = params.test_as_validation, 
                              dtype              = params.dtype)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return dataset
