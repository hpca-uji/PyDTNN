import importlib, sys, numpy as np

def get_dataset(params):
    try:
        dataset_name = {"mnist": "MNIST", "cifar10": "CIFAR10", "imagenet": "ImageNet"}
        dataset_mod = importlib.import_module("datasets.NN_dataset")
        dataset_obj = getattr(dataset_mod, dataset_name[params.dataset])
        dataset = dataset_obj(train_path         = params.dataset_train_path, 
                              test_path          = params.dataset_test_path,
                              model              = params.model,
                              test_as_validation = params.test_as_validation,
                              flip_images        = params.flip_images,
                              flip_images_prob   = params.flip_images_prob,
                              crop_images        = params.crop_images,
                              crop_images_size   = params.crop_images_size,
                              crop_images_prob   = params.crop_images_prob,
                              dtype              = params.dtype,
                              use_synthetic_data = params.use_synthetic_data)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        sys.exit(-1)
    return dataset
