# Python Distributed Training of Neural Networks - PyDTNN

## Introduction

**PyDTNN** is a light-weight library developed at Universitat Jaume I (Spain)
for distributed Deep Learning training and inference that offers an initial starting point for interaction 
with distributed training of (and inference with) deep neural networks. 
PyDTNN priorizes simplicity over efficiency, providing an amiable user 
interface which enables a flat accessing curve. To perform the training and 
inference processes, PyDTNN exploits distributed inter-process parallelism 
(via MPI) for clusters and intra-process (via multi-threading) parallelism 
to leverage the presence of multicore processors at node level. For that, 
PyDTNN uses MPI4Py for message-passing and im2col transforms to cast the 
convolutions in terms of dense matrix-matrix multiplications, 
which are realized BLAS calls via NumPy.

Supported layers:

  * Fully-connected
  * Convolutional 2D
  * Max pooling
  * Dropout

Supported datasets:

  * **MNIST** handwritten digit database: this dataset is included into the project.

  * **ImageNet**: due to memory constraints, the PyDTNN module for this dataset requires 
  a preprocessed ImageNet dataset split into 1,024 files in the NPZ Numpy's compressed 
  array format containing the images/labels. Each of these files should store the images 
  in the key 'x' with the shape NCHW = (1251, 3, 227, 227) and the labels with the shape 
  NL = (1251, 1) in the key 'y'. Images shall be stored in np.uint8 data type in the 
  range [0..255] while the labels can be stored in np.int16 in the range [1..1000].
```
    >>> import numpy as np
    >>> data = np.load("/scratch/imagenet/train/train-00000-of-01024.npz")
    >>> data['x'].shape
    (1251, 3, 227, 227)
    >>> data['y'].shape
    (1251, 1)
```

## Installing PyDTNN

**PyDTNN** requires Python3 with the following packages:
```
    mpi4py==3.0.2
    tqdm==4.43.0
    scipy==1.3.0
    Cython==0.29.13
    numpy==1.17.2
    pycuda==2019.1.2
```

You will also need a MPI library installed, such as OpenMPI, MPICH or MVAPICH2.

To realize the im2col/col2im transforms for Convolutional layers in parallel, 
it is necessary to compile ``NN_im2col_cython.pyx`` Cython module (see ``compile_im2col.sh`` script).
```
LDSHARED="gcc -shared" CC=gcc python3 setup.py build_ext --inplace
```

Note the ``-fopenmp`` compilation flag in the ``setup.py`` file to exploit intra-process parallelism via OpenMP threads.

## Launcher options

PyDTNN framework comes with a utility NN launcher `tests/benchmarks_CNN.py` supporting the following options:

* Model parameters:
    * ``--model``: Neural network model: `simplemlp`, `simplecnn`, `alexnet`, `vgg11`, `vgg16`.
    * ``--dataset``: Dataset to train: `mnist`, `imagenet`.
    * ``--dataset_train_path``: Path to the training dataset.
    * ``--dataset_test_path``: Path to the training dataset.
    * ``--test_as_validation``: Prevent making partitions on training data for training+validation data, use test data for validation. True if specified.
    * ``--batch_size``: Batch size per MPI rank.
    * ``--validation_split``: Split between training and validation data.
    * ``--steps_per_epoch``: Trims the training data depending on the given number of steps per epoch. Default: 0, i.e., do not trim.
    * ``--num_epochs``: Number of epochs to perform. Default value: 1.
    * ``--evaluate``: Evaluate the model before and after training the model. Default: False.

* Optimizer parameters:
    * ``--optimizer``: Optimizers: `SGD`, `SGDMomentum`, `RMSProp`, `Adam`. Default: `SGDMomentum`.
    * ``--learning_rate``: Learning rate. Default: 0.01.
    * ``--decay_rate``: Decay rate for `RMSProp` and `Adam` optimizers. Default: 0.99.
    * ``--epsilon``: Decay rate for `RMSProp` and `Adam` optimizers. Default: 1e-8.
    * ``--momentum``: Decay rate for `SGDMomentum` optimizer. Default: 0.9.
    * ``--loss_func``: List of comma-separated loss functions that Are evaluated on each trained batch: `categorical_accuracy`,`categorical_cross_entropy`,`categorical_hinge`,`categorical_mse`,`categorical_mae`,`regression_mse`,`regression_mae`.

* Parallelization and other performance-related parameters:
    * ``--parallel``: Data parallelization modes: `sequential`, `data`. Default: `sequential`.
    * ``--non_blocking_mpi``: Enable non-blocking MPI primitives.
    * ``--tracing``: Obtain Extrae traces.
    * ``--profile``: Obtain cProfile profiles.
    * ``--enable_gpu``: Enable GPU for matmul operations.
    * ``--dtype``: Dataype to use: `float32`, `float64`.


## Example: distributed training of a CNN for the MNIST dataset

In this example, we train a simple CNN for the MNIST dataset using data
parallelism and 12 MPI ranks each using 2 OpenMP threads.

```
$ export OMP_NUM_THREADS=2
$ mpirun -np 12 \
   python3 -u benchmarks_CNN.py \
         --model=simplecnn \
         --dataset=mnist \
         --dataset_train_path=../datasets/mnist \
         --dataset_test_path=../datasets/mnist \
         --batch_size=64 \
         --validation_split=0.2 \
         --num_epochs=10 \
         --optimizer=Adam \
         --learning_rate=0.01 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --parallel=data \
         --dtype=float32 --evaluate
$ bash run_mpi_mnist.sh
**** Running with 12 processes...
**** Creating simplecnn model...
┌───────┬──────────┬─────────┬───────────────┬─────────────────┬─────────┬─────────┐
│ Layer │   Type   │ #Params │ Output shape  │  Weights shape  │ Padding │ Stride  │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   0   │  Input   │    0    │  (1, 28, 28)  │                 │         │         │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   1   │  Conv2D  │   40    │  (4, 26, 26)  │  (4, 1, 3, 3)   │    0    │    1    │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   2   │   Relu   │    0    │  (4, 26, 26)  │                 │         │         │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   3   │  Conv2D  │   296   │  (8, 24, 24)  │  (8, 4, 3, 3)   │    0    │    1    │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   4   │   Relu   │    0    │  (8, 24, 24)  │                 │         │         │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   5   │  Pool2D  │    0    │  (8, 12, 12)  │                 │    0    │    2    │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   6   │    FC    │ 147584  │    (128,)     │   (1152, 128)   │         │         │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   7   │   Relu   │    0    │    (128,)     │                 │         │         │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   8   │    FC    │  1290   │     (10,)     │    (128, 10)    │         │         │
├───────┼──────────┼─────────┼───────────────┼─────────────────┼─────────┼─────────┤
│   9   │ Softmax  │    0    │     (10,)     │                 │         │         │
└───────┴──────────┴─────────┴───────────────┴─────────────────┴─────────┴─────────┘
**** Parameters:
  model              : simplecnn
  dataset            : mnist
  dataset_train_path : ../datasets/mnist
  dataset_test_path  : ../datasets/mnist
  test_as_validation : False
  batch_size         : 64
  validation_split   : 0.2
  steps_per_epoch    : 0
  num_epochs         : 10
  evaluate           : True
  optimizer          : Adam
  learning_rate      : 0.01
  decay_rate         : 0.99
  epsilon            : 1e-08
  momentum           : 0.9
  loss_func          : categorical_accuracy,categorical_cross_entropy
  parallel           : data
  blocking_mpi       : True
  tracing            : False
  profile            : False
  enable_gpu         : False
  dtype              : float32
**** Loading mnist dataset...
**** Evaluating on test dataset...
Testing: 100%|████████████████████| 10000/10000 [00:01<00:00, 30623.08 samples/s, test_acc: 10.07%, test_cro: 2.3007619]
**** Training...
Epoch  1/10: 100%|████████████████| 48000/48000 [00:01<00:00, 29545.50 samples/s, acc: 82.75%, cro: 0.5958990, val_acc: 91.65%, val_cro: 0.3004644]                                              
Epoch  2/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30469.53 samples/s, acc: 93.29%, cro: 0.2284573, val_acc: 94.69%, val_cro: 0.1804817]                                              
Epoch  3/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30603.47 samples/s, acc: 95.66%, cro: 0.1458783, val_acc: 96.29%, val_cro: 0.1209923]                                              
Epoch  4/10: 100%|████████████████| 48000/48000 [00:01<00:00, 31015.46 samples/s, acc: 96.97%, cro: 0.1014266, val_acc: 96.78%, val_cro: 0.1035603]                                              
Epoch  5/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30793.53 samples/s, acc: 97.52%, cro: 0.0804216, val_acc: 98.02%, val_cro: 0.0664031]                                              
Epoch  6/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30646.87 samples/s, acc: 98.03%, cro: 0.0613005, val_acc: 98.35%, val_cro: 0.0533145]                                              
Epoch  7/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30993.96 samples/s, acc: 98.46%, cro: 0.0499855, val_acc: 98.75%, val_cro: 0.0381323]                                              
Epoch  8/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30864.58 samples/s, acc: 98.78%, cro: 0.0390398, val_acc: 98.91%, val_cro: 0.0345537]                                              
Epoch  9/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30758.84 samples/s, acc: 99.07%, cro: 0.0305507, val_acc: 99.11%, val_cro: 0.0302036]                                              
Epoch 10/10: 100%|████████████████| 48000/48000 [00:01<00:00, 30654.72 samples/s, acc: 99.16%, cro: 0.0266255, val_acc: 99.31%, val_cro: 0.0223465]                                              
**** Done... and thanks for all the fish!!!
Time: 18.34 s
Throughput: 32712.78 samples/s
**** Evaluating on test dataset...
Testing: 100%|████████████████████| 10000/10000 [00:01<00:00, 30423.38 samples/s, test_acc: 98.17%, test_cro: 0.0636805]

```

## References

Publications describing PyDTNN:

* Not yet available.

## Acknowledgments

The **PyDTNN** library has been partially supported by:

* Project TIN2017-82972-R **"Agorithmic Techniques for Energy-Aware and Error-Resilient High Performance Computing"** funded by the Spanish Ministry of Economy and Competitiveness (2018-2020).

* Project RTI2018-098156-B-C51 **"Innovative Technologies of Processors, Accelerators and Networks for Data Centers and High Performance Computing"** funded by the Spanish Ministry of Science, Innocation and Universities.

* Project CDEIGENT/2017/04 **"High Performance Computing for Neural Networks"** funded by the Valencian Government.

* Project UJI-A2019-11 **"Energy-Aware High Performance Computing for Deep Neural Networks"** funded by the Universitat Jaume I.

