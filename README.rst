.. figure:: pydtnn.svg
   :width: 50 %
   
Python Distributed Training of Neural Networks
=======================================================

Introduction
------------

PyDTNN is a light-weight library developed at Universitat Jaume I (Spain) for
distributed Deep Learning training and inference that offers an initial starting
point for interaction with distributed training of (and inference with) deep
neural networks. PyDTNN prioritizes simplicity over efficiency, providing an
amiable user interface which enables a flat accessing curve. To perform the
training and inference processes, PyDTNN exploits distributed inter-process
parallelism (via MPI) for clusters and intra-process (via multi-threading)
parallelism to leverage the presence of multicore processors and GPUs at node
level. For that, PyDTNN uses MPI4Py for message-passing, BLAS calls via NumPy
for multicore processors and PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

Supported layers:

-  Fully-connected
-  Convolutional 2D
-  Max pooling 2D
-  Average pooling 2D
-  Dropout
-  Flatten
-  Batch normalization
-  Addition block (for residual nets, e.g., ResNet)
-  Concatenation block (for channel concatenation-based nets, e.g.,
   Inception, GoogleNet, DenseNet, etc.)

Supported datasets:

-  **MNIST** handwritten digit database. This dataset is included into
   the project.

-  **CIFAR10** database of the 80 million tiny images dataset. This
   dataset is not included into the project. Its binary version can be
   downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

-  **ImageNet**: the PyDTNN module for this dataset requires a
   preprocessed ImageNet dataset split into 1,024 files in the NPZ
   Numpy compressed array format containing the images/labels, similar
   to what TensorFlow uses. Each of these files should store the images
   in the key 'x' with the shape NCHW = (1251, 3, 227, 227), and the
   labels in the key 'y' with the shape NL = (1251, 1). Images shall be
   stored in np.uint8 data type in the range [0..255] while the labels
   can be stored in np.int16 in the range [1..1000]::

    >>> import numpy as np
    >>> data = np.load("/scratch/imagenet/train/train-00000-of-01024.npz")
    >>> data['x'].shape
    (1251, 3, 227, 227)
    >>> data['y'].shape
    (1251, 1)

   PyDTNN comes with the utility ``datasets/ImageNet_converter.py`` that reads the
   preprocessed ImageNet TensorFlow training/validation files in TFRecord format
   and converts them into NPZ format.


Installing PyDTNN from source
-----------------------------

Download PyDTNN source code from its GitHub repository and enter the PyDTNN directory::

    $ git clone https://github.com/hpca-uji/PyDTNN
    $ cd PyDTNN

The required Python packages are listed in the ``requirements.txt`` file, to install
them, type::

    $ pip install -r requirements.txt

Then, the PyDTNN package itself must be installed::

    $ pip install .

If you plan to modify the PyDTNN code, instead of using the previous line, you
can install PyDTNN in editable mode (see ``DEVELOPMENT.rst`` for more details)::

    $ pip install -e .

Optionally, if you are going to use either MPI or CUDA, you should have
installed the corresponding system libraries, and install the required Python
packages with::

    $ pip install -r requirements_mpi.txt       # If MPI is going to be used
    $ pip install -r requirements_cuda_1.txt    # If CUDA is going to be used
    $ pip install -r requirements_cuda_2.txt


Launcher options
----------------

The PyDTNN framework comes with a utility launcher called
``pydtnn_benchmark.py`` that supports the following options:

-  Model parameters:

   -  ``--model``: Neural network model: ``simplemlp``, ``simplecnn``,
      ``alexnet``, ``vgg11``, ``vgg16``, etc.
   -  ``--dataset``: Dataset to train: ``mnist``, ``cifar10``,
      ``imagenet``.
   -  ``--dataset_train_path``: Path to the training dataset.
   -  ``--dataset_test_path``: Path to the training dataset.
   -  ``--tensor_format``: Data format to be used: ``NHWC`` or ``NCHW``.
      Optionally, the ``AUTO`` value sets ``NCHW`` when the option 
      ``--enable_gpu`` is set and ``NHWC`` otherwise. Default: ``AUTO``.
   -  ``--test_as_validation``: Prevent making partitions on training
      data for training+validation data, use test data for validation.
      True if specified.
   -  ``--flip_images``: Enable horizontal flip of images in the
      dataset. Default: False
   -  ``--flip_images_prob``: Probability of horizontal flip of images
      in the dataset. Default: 0.5
   -  ``--crop_images``: Enable random cropping of images in the
      dataset. Default: False
   -  ``--crop_images_prob``: Probability of random cropping of images
      in the dataset. Default: 0.5
   -  ``--batch_size``: Batch size per MPI rank.
   -  ``--validation_split``: Split between training and validation
      data.
   -  ``--steps_per_epoch``: Trims the training data depending on the
      given number of steps per epoch. Default: 0, i.e., do not trim.
   -  ``--num_epochs``: Number of epochs to perform. Default value: 1.
   -  ``--evaluate``: Evaluate the model before and after training the
      model. Default: False.
   -  ``--weights_and_bias_filename``: Load weights and bias from file.
      Default: None.
   -  ``--shared_storage``: If true only rank 0 can dump weights and
      bias onto a file. Default: True.

-  Optimizer parameters:

   -  ``--optimizer``: Optimizers: ``sgd``, ``rmsprop``, ``adam``,
      ``nadam``. Default: ``sgd``.
   -  ``--learning_rate``: Learning rate. Default: 0.01.
   -  ``--learning_rate_scaling``: Scale learning rate in data
      parallelism: new\_lr = lr \* num\_procs.
   -  ``--momentum``: Decay rate for ``sgd`` optimizer. Default: 0.9.
   -  ``--rho``: Variable for ``rmsprop`` optimizers. Default: 0.99.
   -  ``--epsilon``: Variable for ``rmsprop``, ``adam``, ``nadam``
      optimizers. Default: 1e-8.
   -  ``--beta1``: Variable for ``adam``, ``nadam`` optimizers. Default:
      0.99.
   -  ``--beta2``: Variable for ``adam``, ``nadam`` optimizers. Default:
      0.999.
   -  ``--nesterov``: Whether to apply Nesterov momentum. Default:
      False.
   -  ``--loss_func``: Loss functions that is evaluated on each trained
      batch: ``categorical_cross_entropy``, ``binary_cross_entropy``.
   -  ``--metrics``: List of comma-separated metrics that are evaluated
      on each trained batch:
      ``categorical_accuracy``,\ ``categorical_hinge``,\ ``categorical_mse``,\ ``categorical_mae``,\ ``regression_mse``,\ ``regression_mae``.

-  Learning rate schedulers parameters:

   -  ``--lr_schedulers``: List of comma-separated LR schedulers:
      ``warm_up``, ``early_stopping``, ``reduce_lr_on_plateau``,
      ``reduce_lr_every_nepochs``, ``model_checkpoint``
   -  ``--warm_up_batches``: Number of batches (ramp up) that the LR is
      scaled up from 0 until LR.
   -  ``--early_stopping_metric``: Loss metric monitored by
      early\_stopping LR scheduler.
   -  ``--early_stopping_patience``: Number of epochs with no
      improvement after which training will be stopped.
   -  ``--reduce_lr_on_plateau_metric``: Loss metric monitored by
      reduce\_lr\_on\_plateau LR scheduler.
   -  ``--reduce_lr_on_plateau_factor``: Factor by which the learning
      rate will be reduced. new\_lr = lr \* factor.
   -  ``--reduce_lr_on_plateau_patience``: Number of epochs with no
      improvement after which LR will be reduced.
   -  ``--reduce_lr_on_plateau_min_lr``: Lower bound on the learning
      rate.
   -  ``--reduce_lr_every_nepochs_factor``: Factor by which the learning
      rate will be reduced. new\_lr = lr \* factor.
   -  ``--reduce_lr_every_nepochs_nepochs``: Number of epochs after
      which LR will be periodically reduced.
   -  ``--reduce_lr_every_nepochs_min_lr``: Lower bound on the learning
      rate.
   -  ``--model_checkpoint_metric``: Loss metric monitored by
      model\_checkpoint LR scheduler.
   -  ``--model_checkpoint_save_freq``: Frequency (in epochs) at which
      the model weights and bias will be saved by the model\_checkpoint
      LR scheduler.

-  Parallelization and other performance-related parameters:

   -  ``--parallel``: Data parallelization modes: ``sequential``,
      ``data``. Default: ``sequential``.
   -  ``--non_blocking_mpi``: Enable non-blocking MPI primitives.
   -  ``--enable_gpu``: Enable GPU, use cuDNN library.
   -  ``--enable_gpudirect``: Enable GPU pinned memory for gradients
      when using a CUDA-aware MPI version.
   -  ``--enable_cudnn_auto_conv_alg``: Let cuDNN to select the best
      performing convolution algorithm.
   -  ``--enable_nccl``: Enable the use of the NCCL library for 
      collective communications on GPUs. This option can only be set if 
      with ``--enable_gpu``.
   -  ``--enable_conv_gemm``: Enables the use of libconvGemm to replace
      im2col and gemm operations.
   -  ``--dtype``: Datatype to use: ``float32``, ``float64``.

-  Tracing and profiling parameters:

   -  ``--tracing``: Obtain Simple/Extrae-based traces.
   -  ``--tracer_output``: Output file to store the Simple/Extrae-based 
     traces.
   -  ``--profile``: Obtain cProfile profiles.

Example: distributed training of a CNN for the MNIST dataset
------------------------------------------------------------

In this example, we train a simple CNN for the MNIST dataset using data
parallelism and 12 MPI ranks each using 4 OpenMP threads::

    $ export OMP_NUM_THREADS=4
    $ mpirun -np 12 \
        python3 -Ou pydtnn_benchmark.py \
          --model=simplecnn \
          --dataset=mnist \
          --dataset_train_path=datasets/mnist \
          --dataset_test_path=datasets/mnist \
          --test_as_validation=False \
          --flip_images=True \
          --batch_size=64 \
          --validation_split=0.2 \
          --num_epochs=50 \
          --evaluate=True \
          --optimizer=adam \
          --learning_rate=0.01 \
          --loss_func=categorical_cross_entropy \
          --lr_schedulers=warm_up,reduce_lr_every_nepochs \
          --reduce_lr_every_nepochs_factor=0.5 \
          --reduce_lr_every_nepochs_nepochs=30 \
          --reduce_lr_every_nepochs_min_lr=0.001 \
          --early_stopping_metric=val_categorical_cross_entropy \
          --early_stopping_patience=20 \
          --parallel=sequential \
          --tracing=False \
          --profile=False \
          --enable_gpu=True \
          --dtype=float32


    **** simplecnn model...
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    | Layer |           Type           | #Params | Output shape  |   Weights shape   |       Parameters       |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   0   |          Input           |    0    |  (1, 28, 28)  |                   |                        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   1   |          Conv2D          |   40    |  (4, 28, 28)  |   (4, 1, 3, 3)    |padd=(1,1), stride=(1,1)|
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   2   |          Conv2D          |   148   |  (4, 28, 28)  |   (4, 4, 3, 3)    |padd=(1,1), stride=(1,1)|
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   3   |        MaxPool2D         |    0    |  (4, 14, 14)  |      (2, 2)       |padd=(0,0), stride=(2,2)|
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   4   |         Flatten          |    0    |    (784,)     |                   |                        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   5   |            FC            | 100480  |    (128,)     |    (784, 128)     |                        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   6   |           Relu           |    0    |    (128,)     |                   |                        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   7   |         Dropout          |    0    |    (128,)     |                   |       rate=0.50        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   8   |            FC            |  1290   |     (10,)     |     (128, 10)     |                        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |   9   |         Softmax          |    0    |     (10,)     |                   |                        |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    |             Total parameters       101958    398.27 KBytes                                              |
    +-------+--------------------------+---------+---------------+-------------------+------------------------+
    **** Loading mnist dataset...
    **** Parameters:
      model                          : simplecnn
      dataset                        : mnist
      dataset_train_path             : datasets/mnist
      dataset_test_path              : datasets/mnist
      test_as_validation             : False
      flip_images                    : True
      flip_images_prob               : 0.5
      crop_images                    : False
      crop_images_size               : 16
      crop_images_prob               : 0.5
      batch_size                     : 64
      global_batch_size              : None
      validation_split               : 0.2
      steps_per_epoch                : 0
      num_epochs                     : 50
      evaluate                       : True
      weights_and_bias_filename      : None
      shared_storage                 : True
      history_file                   : None
      optimizer                      : adam
      learning_rate                  : 0.01
      learning_rate_scaling          : True
      momentum                       : 0.9
      decay                          : 0.0
      nesterov                       : False
      beta1                          : 0.99
      beta2                          : 0.999
      epsilon                        : 1e-07
      rho                            : 0.9
      loss_func                      : categorical_cross_entropy
      metrics                        : categorical_accuracy
      lr_schedulers                  : warm_up,reduce_lr_every_nepochs
      warm_up_epochs                 : 5
      early_stopping_metric          : val_categorical_cross_entropy
      early_stopping_patience        : 20
      reduce_lr_on_plateau_metric    : val_categorical_cross_entropy
      reduce_lr_on_plateau_factor    : 0.1
      reduce_lr_on_plateau_patience  : 5
      reduce_lr_on_plateau_min_lr    : 0
      reduce_lr_every_nepochs_factor : 0.5
      reduce_lr_every_nepochs_nepochs: 30
      reduce_lr_every_nepochs_min_lr : 0.001
      stop_at_loss_metric            : val_accuracy
      stop_at_loss_threshold         : 0
      model_checkpoint_metric        : val_categorical_cross_entropy
      model_checkpoint_save_freq     : 2
      mpi_processes                  : 12
      threads_per_process            : 4
      parallel                       : data
      non_blocking_mpi               : False
      tracing                        : False
      profile                        : False
      gpus_per_node                  : 0
      enable_conv_gemm               : False
      enable_gpu                     : False
      enable_gpudirect               : False
      enable_nccl                    : False
      dtype                          : float32
    **** Evaluating on test dataset...
    Testing: 100%|████████████████████| 10000/10000 [00:00<00:00, 29732.29 samples/s, test_acc: 12.50%, test_cro: 2.3008704]
    **** Training...
    Epoch  1/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11184.77 samples/s, acc: 71.35%, cro: 1.2238941, val_acc: 88.49%, val_cro: 0.4369879]
    Epoch  2/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10691.66 samples/s, acc: 88.87%, cro: 0.4051699, val_acc: 91.10%, val_cro: 0.3070377]
    Epoch  3/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10617.42 samples/s, acc: 90.98%, cro: 0.3086980, val_acc: 92.56%, val_cro: 0.2624177]
    Epoch  4/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10874.49 samples/s, acc: 92.43%, cro: 0.2576146, val_acc: 93.83%, val_cro: 0.2232232]
    Epoch  5/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10451.37 samples/s, acc: 93.48%, cro: 0.2159374, val_acc: 94.76%, val_cro: 0.1868786]
    Epoch  6/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10756.92 samples/s, acc: 94.81%, cro: 0.1748247, val_acc: 95.63%, val_cro: 0.1544418]
    Epoch  7/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10901.69 samples/s, acc: 95.77%, cro: 0.1417673, val_acc: 96.25%, val_cro: 0.1331401]
    Epoch  8/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11046.87 samples/s, acc: 96.55%, cro: 0.1164078, val_acc: 96.80%, val_cro: 0.1134956]
    Epoch  9/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10944.06 samples/s, acc: 97.05%, cro: 0.0992564, val_acc: 96.98%, val_cro: 0.1033213]
    Epoch 10/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11017.46 samples/s, acc: 97.48%, cro: 0.0866701, val_acc: 97.28%, val_cro: 0.0972526]
    Epoch 11/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10871.28 samples/s, acc: 97.67%, cro: 0.0769905, val_acc: 97.58%, val_cro: 0.0862264]
    Epoch 12/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10982.79 samples/s, acc: 97.99%, cro: 0.0682642, val_acc: 97.55%, val_cro: 0.0828536]
    Epoch 13/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11115.45 samples/s, acc: 98.16%, cro: 0.0616423, val_acc: 97.77%, val_cro: 0.0782390]
    Epoch 14/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10783.11 samples/s, acc: 98.30%, cro: 0.0562393, val_acc: 97.91%, val_cro: 0.0716845]
    Epoch 15/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10642.71 samples/s, acc: 98.49%, cro: 0.0515601, val_acc: 97.93%, val_cro: 0.0696817]
    Epoch 16/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10643.74 samples/s, acc: 98.62%, cro: 0.0468920, val_acc: 97.98%, val_cro: 0.0688842]
    Epoch 17/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10726.90 samples/s, acc: 98.70%, cro: 0.0434075, val_acc: 98.10%, val_cro: 0.0675637]
    Epoch 18/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10558.22 samples/s, acc: 98.71%, cro: 0.0424472, val_acc: 98.25%, val_cro: 0.0641221]
    Epoch 19/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10809.42 samples/s, acc: 98.86%, cro: 0.0382850, val_acc: 98.19%, val_cro: 0.0646157]
    Epoch 20/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10712.11 samples/s, acc: 98.95%, cro: 0.0348660, val_acc: 98.25%, val_cro: 0.0617139]
    Epoch 21/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11068.46 samples/s, acc: 99.05%, cro: 0.0323043, val_acc: 98.14%, val_cro: 0.0658118]
    Epoch 22/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11436.88 samples/s, acc: 99.06%, cro: 0.0306285, val_acc: 98.17%, val_cro: 0.0648578]
    Epoch 23/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11093.08 samples/s, acc: 99.17%, cro: 0.0282567, val_acc: 98.22%, val_cro: 0.0661603]
    Epoch 24/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11058.23 samples/s, acc: 99.14%, cro: 0.0275220, val_acc: 98.28%, val_cro: 0.0638472]
    Epoch 25/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11362.12 samples/s, acc: 99.27%, cro: 0.0242397, val_acc: 98.32%, val_cro: 0.0616558]
    Epoch 26/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10929.57 samples/s, acc: 99.33%, cro: 0.0228250, val_acc: 98.41%, val_cro: 0.0614293]
    Epoch 27/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10582.72 samples/s, acc: 99.33%, cro: 0.0218627, val_acc: 98.30%, val_cro: 0.0647660]
    Epoch 28/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11540.73 samples/s, acc: 99.40%, cro: 0.0202375, val_acc: 98.31%, val_cro: 0.0653990]
    Epoch 29/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11089.71 samples/s, acc: 99.47%, cro: 0.0187735, val_acc: 98.33%, val_cro: 0.0642570]
    Epoch 30/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11112.27 samples/s, acc: 99.51%, cro: 0.0166023, val_acc: 98.40%, val_cro: 0.0630408]
    Epoch 31/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11004.81 samples/s, acc: 99.56%, cro: 0.0154129, val_acc: 98.24%, val_cro: 0.0669048]
    LRScheduler ReduceLROnPlateau: metric val_categorical_cross_entropy did not improve for 5 epochs, setting learning rate to 0.01000000
    Epoch 32/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11015.29 samples/s, acc: 99.70%, cro: 0.0122010, val_acc: 98.39%, val_cro: 0.0635789]
    Epoch 33/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11166.31 samples/s, acc: 99.74%, cro: 0.0111252, val_acc: 98.44%, val_cro: 0.0624000]
    Epoch 34/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11112.63 samples/s, acc: 99.74%, cro: 0.0108013, val_acc: 98.44%, val_cro: 0.0627380]
    Epoch 35/50: 100%|████████████████| 48000/48000 [00:04<00:00, 10914.84 samples/s, acc: 99.76%, cro: 0.0105415, val_acc: 98.47%, val_cro: 0.0627000]
    Epoch 36/50: 100%|████████████████| 48000/48000 [00:04<00:00, 11017.57 samples/s, acc: 99.76%, cro: 0.0103665, val_acc: 98.50%, val_cro: 0.0628462]
    LRScheduler EarlyStopping: metric val_categorical_cross_entropy did not improve for 10 epochs, stop training!
    LRScheduler ReduceLROnPlateau: metric val_categorical_cross_entropy did not improve for 5 epochs, setting learning rate to 0.00100000
    **** Done...
    Time: 173.59 s
    Throughput: 17282.50 samples/s
    **** Evaluating on test dataset...
    Testing: 100%|███████████████████| 10000/10000 [00:00<00:00, 28720.12 samples/s, test_acc: 100.00%, test_cro: 0.0000443]


Citing PyDTNN
-------------

If you use PyDTNN, and you would like to acknowledge the project
in your academic publication, we suggest citing the following paper:

-  **PyDTNN: A user-friendly and extensible framework for distributed
   deep learning**. Sergio Barrachina, Adrián Castelló, Mar Catalán,
   Manuel F. Dolz, Jose I. Mestre. *Journal of Supercomputing*. ISSN:
   1573-0484. DOI: `10.1007/s11227-021-03673-z
   <http://dx.doi.org/10.1007/s11227-021-03673-z>`_.

Other references:

-  **A Flexible Research-Oriented Framework for Distributed Training 
   of Deep Neural Networks**. Sergio Barrachina, Adrián Castelló, 
   Mar Catalán, Manuel F. Dolz and Jose I. Mestre. *2021 IEEE 
   International Parallel and Distributed Processing Symposium 
   Workshops (IPDPSW)*, 2021, pp. TBD, DOI: `TDB 
   <http://dx.doi.org/10.1007/TBD>`_.


Acknowledgments
---------------

The PyDTNN library has been partially supported by:

-  Project TIN2017-82972-R **"Algorithmic Techniques for Energy-Aware and
   Error-Resilient High Performance Computing"** funded by the Spanish
   Ministry of Economy and Competitiveness (2018-2020).

-  Project RTI2018-098156-B-C51 **"Innovative Technologies of
   Processors, Accelerators and Networks for Data Centers and High
   Performance Computing"** funded by the Spanish Ministry of Science,
   Innovation and Universities.

-  Project CDEIGENT/2017/04 **"High Performance Computing for Neural
   Networks"** funded by the Valencian Government.

-  Project UJI-A2019-11 **"Energy-Aware High Performance Computing for
   Deep Neural Networks"** funded by the Universitat Jaume I.
