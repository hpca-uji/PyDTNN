import datasets.MNIST_basics
#import datasets.ImageNet_basics
import numpy as np

def read_dataset(dataset):
    if dataset == "imagenet":
        xall = np.load("../datasets/imagenet/imagenet_xall.npy").astype(np.float64)
        yall = np.load("../datasets/imagenet/imagenet_yall.npy").astype(np.float64)
        #xall = np.tile(xall, 10)
        #yall = np.tile(yall, 10)
        # Random values
        #xall = np.random.random((227, 227, 3, 3072))
        #zall = np.random.randint(1000, size=(3072))
  
        # Real ImageNet data, but needs tensorflow module
        #zall = np.random.randint(1000, size=(3072))
        #xall, zall = ImageNet_basics.load_tfrecords("imagenet/train-00000-of-01024", 224, 3000)
        #yall = np.zeros([1000, zall.shape[0]])
        #for k in range(xall.shape[-1]):
        #   yall[zall[k], k] = 1
  
    elif dataset == "mnist":
        training_dataset = MNIST_basics.read_idx('../datasets/mnist/train_images_idx3_ubyte')
        training_labels = MNIST_basics.read_idx('../datasets/mnist/train_labels_idx1_ubyte')
        train_dataset_size = 60000
        xall = np.transpose(training_dataset.reshape(train_dataset_size, 1, 28, 28))
        zall = np.transpose(training_labels.reshape(train_dataset_size, 1))
        yall = np.zeros([10, train_dataset_size])
        
        for k in range(train_dataset_size):
          yall[zall[0, k], k] = 1
      
    else:
        print("Dataset %s not found!" % dataset)
        sys.exit(-1)

    return xall, yall
