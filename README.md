# Python Distributed Training of Neural Networks
![](logo.svg)

## Introduction
PyDTNN is a light-weight library developed at Universitat Jaume I
(Spain) for distributed Deep Learning training and inference that offers
an initial starting point for interaction with distributed training of
(and inference with) deep neural networks. PyDTNN prioritizes simplicity
over efficiency, providing an amiable user interface which enables a
flat accessing curve. To perform the training and inference processes,
PyDTNN exploits distributed inter-process parallelism (via MPI) for
clusters and intra-process (via multi-threading) parallelism to leverage
the presence of multicore processors and GPUs at node level. For that,
PyDTNN uses mpi4py/pympi/NCCL for message-passing, BLAS calls via
NumPy/Cython for multicore processors and PyCUDA/cuDNN/cuBLAS for NVIDIA
GPUs.

Supported layers:
- Fully-connected
- Convolutional 2D
- Max pooling 2D
- Average pooling 2D
- Dropout
- Flatten
- Batch normalization
- Addition block (for residual nets, e.g., ResNet)
- Concatenation block (for channel concatenation-based nets, e.g.,
  Inception, GoogleNet, DenseNet, etc.)

Supported datasets:
- **MNIST**: handwritten digit database. This dataset is included into
  the repository.
- **CIFAR10**: database of the 80 million tiny images dataset. This
  dataset is not included into the repository. Its binary version can be
  downloaded from: <https://www.cs.toronto.edu/~kriz/cifar.html>
- **ImageNet**: the most highly-used subset of ImageNet is the ImageNet
  Large Scale Visual Recognition Challenge (ILSVRC) 2012-2017 image
  classification and localization dataset. This dataset spans 1000
  object classes and contains 1,281, 167 training images, 50,000
  validation images and 100,000 test images. This dataset is not
  included into the repository. It can be downloaded from:
  <https://image-net.org/challenges/LSVRC/2012/2012-downloads.php>
- **ChestXray**: the NIH Chest X-ray dataset consists of 100,000
  de-identified images of chest x-rays. The images are in PNG format. It
  can be downloaded from: <https://nihcc.app.box.com/v/ChestXray-NIHCC>

## Installing PyDTNN from source
Download PyDTNN source code from its GitHub repository and enter the
PyDTNN directory:
```sh
git clone https://github.com/hpca-uji/PyDTNN.git
cd PyDTNN
```

Then package itself must be installed:
```sh
pip install .
```

If you plan to modify the PyDTNN code, instead of using the previous
line, you can install PyDTNN in editable mode (see `CONTRIBUTING.md` for
more details):
```sh
pip install --config-settings editable_mode=compat -e .
```

Optionally, if you are going to use MPI, you should have installed the
corresponding system libraries, and install the required Python packages
with:
```sh
git submodule update --init vendor/net-queue
pip install ./vendor/net-queue

git submodule update --init vendor/pympi
pip install ./vendor/pympi

pip install .[mpi]
```

Optionally, if you are going to use CUDA, you should have installed the
corresponding system libraries, and install the required Python packages
with:
```sh
pip install nvidia-pyindex
pip install .[cuda]
```

Optionally, if you are going to use FHE, you should have installed the
corresponding system libraries, and install the required Python packages
with:
```sh
pip install .[fhe]
```

Optionally, if you are going to use MPI/TCP, you should enable the protocol
with:
```sh
export PYMPI_PROTO=tcp
```

Optionally, if you are going to use MPI/gRPC, you should enable the protocol
with:
```sh
export PYMPI_PROTO=grpc
```

Optionally, if you are going to use MPI/MQTT, you should have installed a
MQTT broker server, you should enable the protocol with:
```sh
export PYMPI_PROTO=mqtt
```

Optionally, if you are going to use MPI/SSL, you should enable the
transport with:
```sh
export PYMPI_SSL=yes
export PYMPI_SSL_KEY=comms/ssl/key.pem    # server private key
export PYMPI_SSL_CERT=comms/ssl/cert.pem  # server ceritficate
```

For more information on how to manage external dependencies see
`vendor/README.md`.

## Launcher options
The PyDTNN framework comes with a utility launcher called
`pydtnn-benchmark` that supports the following options:

- Model parameters:
  - `--model`: Neural network model: `simplemlp`, `simplecnn`,
    `alexnet`, `vgg11`, `vgg16`, etc. Default: `None`.
  - `--batch-size`: Batch size per MPI rank. Default: `None`.
  - `--global-batch-size`: Batch size between all MPI ranks. Default:
    `None`.
  - `--dtype`: Datatype to use: `float32`, `float64`. Default:
    `float32`.
  - `--num-epochs`: Number of epochs to perform. Default: `1`.
  - `--steps-per-epoch`: Trims the training data depending on the given
    number of steps per epoch. Default: `0`, i.e., do not trim.
  - `--evaluate`: Evaluate the model before and after training the
    model. Default: `False`.
  - `--evaluate-only`: Only evaluate the model. Default: `False`.
  - `--weights-and-bias-filename`: Load weights and bias from file.
    Default: `None`.
  - `--history-file`: Filename to save training loss and metrics.
  - `--shared-storage`: If `True` ranks assume they share the file
    system. Default: `True`.
  - `--model-sync-freq`: Number of batches between model syncronization.
    The `0` value syncronizes gradients every batch. Positive values
    syncronizes gradients and weights every N batches. Negative values
    disables syncronization. Default: `0`.
  - `--model-sync-alg`: Aggregation method used to syncronize models:
    `avg`, `wavg` or `invwavg`. Default: `avg`.
  - `--model-sync-participation`: Rank participation to syncronize
    models: `all` or `avail2all`. Default: `all`.
  - `--model-sync-min-avail`: Minumun ranks with data required to
    syncronize models. Default: `0`.
  - `--initial-model-sync`: Sincronize models on training start.
    Default: `True`.
  - `--final-model-sync`: Sincronize models on training end. Default:
    `True`.
  - `--tensor-format`: Data format to be used: `NHWC` or `NCHW`.
    Optionally, the `AUTO` value sets `NCHW` when the option
    `--enable-gpu` is set and `NHWC` otherwise. Default: `NHWC`.
- Dataset parameters:
  - `--dataset`: Dataset to train: `mnist`, `cifar10`, `cyclone`,
    `tsunamis`, `imagenet`, `archive`, `folder`, `chestxray` or
    `synthetic`. Default: `None`.
  - `--dataset-path`: Path to dataset folder.
  - `--dataset-lang`: Dataset language. Default: `en`.
  - `--dataset-lang2`: Dataset second language. Default: `de`.
  - `--synthetic-train-samples`: Number of synthetic train sample.
    Default: `1000`.
  - `--synthetic-test-samples`: Number of synthetic train sample.
    Default: `100`.
  - `--synthetic-input-shape`: Number of synthetic input shape (coma
    separated). Default: `3,32,32`.
  - `--synthetic-output-shape`: Number of synthetic output shape (coma
    separated). Default: `10`.
  - `--dataset-percentage`: Percentage of dataset that will be used. If
    it is `0`: it is deactivated; if is is a value below `1` (and above
    `0`): it will perform undersampling; and if is is a value above `1`:
    it will perform oversampling. Default: `0`.
  - `--test-as-validation`: Prevent making partitions on training data
    for training+validation data, use test data for validation. `True`
    if specified.
  - `--validation-split`: Split between training and validation data.
  - `--augment-flip`: Flip horizontally training images. Default:
    `False`.
  - `--augment-flip-prob`: Probability to flip training images. Default:
    `0.5`.
  - `--augment-crop`: Crop training images. Default: `False`.
  - `--augment-crop-size`: Size to crop training images. Default: `16`.
  - `--augment-crop-prob`: Probability to crop training images. Default:
    `0.5`.
  - `--validation-split`: Split between training and validation data.
  - `--transform-crop`: Crop the images. `True` if specified.
  - `--transform-crop-perc`: Central crop of the images. Default:
    `0.875`.
  - `--transform-resize`: Resize the images. `True` if specified.
  - `--transform-resize-size`: New size of the images. Default: `300`.
  - `--normalize`: Normalize dataset. Default: `False`.
  - `--normalize-offset`: Offset samples by a value. Default: `-0.45`.
  - `--normalize-scale`: Scale samples by a value. Default: `3.75`.
- Optimization parameters:
  - `--enable-best-of`: Enable the `BestOf` auto-tuner.
  - `--enable-memory-cache`: Enable the memory cache module to use
    persistent memory.
  - `--enable-fused-bn-relu`: Fuse `BatchNormalization` and `Relu`
    layers. `True` if specified.
  - `--enable-fused-conv-relu`: Fuse `Conv2D` and `Relu` layers. `True`
    if specified.
  - `--enable-fused-conv-bn`: Fuse `Conv2D` and `BatchNormalization`
    layers. `True` if specified.
  - `--enable-fused-conv-bn-relu`: Fuse `Conv2D` and
    `BatchNormalization` and `Relu` layers. Default: `False`.
- Convolution operation parameters:
  - `--conv-variant`:Select the standard 2D Convolutional module.
    Options:
    - `i2c` (default): Use the ConvI2C algorithm.
    - `gemm`: Use the ConvGemm algorithm.
    - `winograd`: Use the CondWinograd algorithm.
    - `direct`: Use the ConvDirect algorithm.
  - `--conv-direct-method`: The `ConvDirect` module to realize
    convolutions in `Conv2D` layers.
  - `--conv-direct-method`: Use `ConvDirect` module to realize
    convolutions in `Conv2D` layers. `True` if specified.
  - `--conv-direct-methods-for-best-of`: `ConvDirect` modules to compare
    in `best_of` option if specified.
- Optimizer parameters:
  - `--optimizer`: Optimizers: `sgd`, `rmsprop`, `adam`, `nadam`.
    Default: `sgd`.
  - `--learning-rate`: Learning rate. Default: `0.01`.
  - `--learning-rate-scaling`: Scale learning rate in data parallelism:
    `new_lr = lr/num_procs`. `True` if specified.
  - `--optimizer-momentum`: Decay rate for `sgd` optimizer. Default:
    `0.9`. optimizers. Default: `1e-8`.
  - `--optimizer-decay`: Decay rate for optimizers. Default: `0.0`.
  - `--optimizer-nesterov`: Whether to apply Nesterov momentum. Default:
    `False`.
  - `--optimizer-beta1`: Variable for `adam`, `nadam` optimizers.
    Default: `0.99`.
  - `--optimizer-beta2`: Variable for `adam`, `nadam` optimizers.
    Default: `0.999`.
  - `--optimizer-epsilon`: Variable for `rmsprop`, `adam`, `nadam`.
    Default: `1e-7`.
  - `--optimizer-rho`: Variable for `rmsprop` optimizers. Default:
    `0.99`.
  - `--loss-func`: Loss functions that is evaluated on each trained
    batch: `categorical_cross_entropy`, `binary_cross_entropy`. Default
    `categorical_cross_entropy`.
  - `--metrics`: List of comma-separated metrics that are evaluated on
    each trained batch: `categorical_accuracy`, `categorical_hinge`,
    `categorical_mse`, `categorical_mae`, `regression_mse`,
    `regression_mae`, `binary_confusion_matrix`,
    `multiclass_confusion_matrix`, `precision`, `recall`, `f1_score`.
    Default: `categorical_accuracy`.
- Learning rate schedulers parameters:
  - `--schedulers`: List of comma-separated LR schedulers: `warm_up`,
    `early_stopping`, `reduce_lr_on_plateau`, `reduce_lr_every_nepochs`,
    `model_checkpoint`. Default:
    `early_stopping,reduce_lr_on_plateau,model_checkpoint`.
  - `--warm-up-batches`: Number of batches (ramp up) that the LR is
    scaled up from 0 until LR. Default: `5`.
  - `--early-stopping-metric`: Loss metric monitored by `early_stopping`
    scheduler. Default: `val_categorical_cross_entropy`.
  - `--early-stopping-patience`: Number of epochs with no improvement
    after which training will be stopped. Default: `10`.
  - `--early-stopping-minimize`: Whether to minize the metric. If False,
    it will maximize. Default: `True`.
  - `--reduce-lr-on-plateau-metric`: Loss metric monitored by
    `reduce_lr_on_plateau` scheduler. Default:
    `val_categorical_cross_entropy`.
  - `--reduce-lr-on-plateau-factor`: Factor by which the learning rate
    will be reduced. `new_lr = lr *factor`. Default: `0.1`.
  - `--reduce-lr-on-plateau-patience`: Number of epochs with no
    improvement after which LR will be reduced. Default: `5`.
  - `--reduce-lr-on-plateau-min-lr`: Lower bound on the learning rate.
    Default: `0`.
  - `--reduce-lr-every-nepochs-factor`: Factor by which the learning
    rate will be reduced. `new_lr = lr*factor`. Default: `0.1`.
  - `--reduce-lr-every-nepochs-nepochs`: Number of epochs after which LR
    will be periodically reduced. Default: `5`.
  - `--reduce-lr-every-nepochs-min-lr`: Lower bound on the learning
    rate. Default: `0`.
  - `--stop-at-loss-metric`: Loss metric monitored by `stop_at_loss`
    scheduler. Default: `val_accuracy`.
  - `--stop-at-loss-threshold`: Metric threshold monitored by
    `stop_at_loss` scheduler. Default: `0`.
  - `--model-checkpoint-metric`: Loss metric monitored by
    `model_checkpoint` scheduler. Default:
    `val_categorical_cross_entropy`
  - `--model-checkpoint-save-freq`: Frequency (in epochs) at which the
    model weights and bias will be saved by the `model_checkpoint`
    scheduler. Default: `2`.
- Parallelization and other performance-related parameters:
  - `--parallel`: Data parallelization modes: `sequential`, `data`
    (MPI). Default: `sequential`.
  - `--use-blocking-mpi`: Enable blocking MPI primitives. Default:
    `True`.
  - `--use-mpi-buffers`: Enable the use of MPI buffers. Possible values:
    `True` (MPI operations by buffer), `False` (MPI operations by
    object) or `None` (auto-select the better option). Default: `None`.
  - `--enable-gpu`: Enable GPU, use `cuDNN` library. Default: `False`.
  - `--enable-gpudirect`: Enable GPU pinned memory for gradients when
    using a CUDA-aware MPI version. Default: `False`.
  - `--enable-nccl`: Enable the use of the `NCCL` library for collective
    communications on GPUs. This option can only be set with
    `--enable-gpu`. Default. `False`.
  - `--enable-cudnn-auto-conv-alg`: Let `cuDNN` to select the best
    performing convolution algorithm. Default: `True`.
- Encryption parameters:
  - `--encryption`: Encryption library: `tenseal`, `openfhe`, `None`.
    Default `None`.
  - `--encryption-poly-degree`: Encryption polynomial degree.
    `2 ^ value`. Default: `13`.
  - `--encryption-global-scale`: Encryption global scale. `2 ^ value`.
    Default: `40`.
  - `--encryption-security-level`: Encryption security level: `0` (Not
    set), `128`, `192`, `256`. Default: `128`.
- Tracing and profiling parameters:
  - `--tracing`: Obtain Simple/Extrae-based traces. Default: `False`.
  - `--tracer-output`: Output file to store the Simple/Extrae-based
    traces.
  - `--tracer-pmlib-server`: Address of PMlib tracer server. Default:
    `127.0.0.1`.
  - `--tracer-pmlib-port`: Port of PMlib tracer server. Default: `6526`.
  - `--tracer-pmlib-device`: Port of PMlib tracer device.
  - `--profile`: Obtain cProfile profiles. Default: `False`.

## Example: distributed training of a CNN for the MNIST dataset
In this example, we train a simple CNN for the MNIST dataset using data
parallelism and 12 MPI ranks each using 4 OpenMP threads:
```
$ export OMP_NUM_THREADS=4
$ mpirun -np 12 \
    pydtnn-benchmark \
      --model=simplecnn \
      --dataset=mnist \
      --dataset-path=datasets/mnist \
      --dataset-train-path=datasets/mnist \
      --dataset-test-path=datasets/mnist \
      --test-as-validation=False \
      --augment-flip=True \
      --batch-size=64 \
      --validation-split=0.2 \
      --num-epochs=50 \
      --evaluate=True \
      --optimizer=adam \
      --learning-rate=0.01 \
      --loss-func=categorical_cross_entropy \
      --schedulers=warm_up,reduce_lr_every_nepochs \
      --reduce-lr-every-nepochs-factor=0.5 \
      --reduce-lr-every-nepochs-nepochs=30 \
      --reduce-lr-every-nepochs-min-lr=0.001 \
      --early-stopping-metric=val_categorical_cross_entropy \
      --early-stopping-patience=20 \
      --parallel=sequential \
      --tracing=False \
      --profile=False \
      --enable-gpu=True \
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
  augment_flip                    : True
  augment_flip_prob               : 0.5
  augment_crop                    : False
  augment_crop_size               : 16
  augment_crop_prob               : 0.5
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
  schedulers                  : warm_up,reduce_lr_every_nepochs
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
```

## Example: inference of the VGG16 CNN for the CIFAR-10 dataset
In this example, we perform inference with the CNN VGG16 for the
CIFAR-10 dataset using 4 OpenMP threads:
```
$ export OMP_NUM_THREADS=4
$ pydtnn-benchmark \
    --model=vgg16_cifar10 \
    --dataset=cifar10 \
    --dataset-path=datasets/cifar10/cifar-10-binary.tar.gz \
    --evaluate-only=True \
    --batch-size=64 \
    --validation-split=0.2 \
    --weights-and-bias-filename=vgg16-weights-nhwc.npz \
    --tracing=False \
    --profile=False \
    --enable-gpu=True \
    --dtype=float32


**** vgg16_cifar10 model...
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
| Layer |           Type           | #Params | Output shape  |   Weights shape   |             Parameters              |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   0   |         InputCPU         |    0    |  (32, 32, 3)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   1   |        Conv2DCPU         |  1792   | (32, 32, 64)  |   (3, 3, 3, 64)   |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   2   |         ReluCPU          |    0    | (32, 32, 64)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   3   |        Conv2DCPU         |  36928  | (32, 32, 64)  |  (64, 3, 3, 64)   |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   4   |         ReluCPU          |    0    | (32, 32, 64)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   5   |       MaxPool2DCPU       |    0    | (16, 16, 64)  |      (2, 2)       |padd=(0,0), stride=(2,2), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   6   |        Conv2DCPU         |  73856  | (16, 16, 128) |  (64, 3, 3, 128)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   7   |         ReluCPU          |    0    | (16, 16, 128) |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   8   |        Conv2DCPU         | 147584  | (16, 16, 128) | (128, 3, 3, 128)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|   9   |         ReluCPU          |    0    | (16, 16, 128) |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  10   |       MaxPool2DCPU       |    0    |  (8, 8, 128)  |      (2, 2)       |padd=(0,0), stride=(2,2), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  11   |        Conv2DCPU         | 295168  |  (8, 8, 256)  | (128, 3, 3, 256)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  12   |         ReluCPU          |    0    |  (8, 8, 256)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  13   |        Conv2DCPU         | 590080  |  (8, 8, 256)  | (256, 3, 3, 256)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  14   |         ReluCPU          |    0    |  (8, 8, 256)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  15   |        Conv2DCPU         | 590080  |  (8, 8, 256)  | (256, 3, 3, 256)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  16   |         ReluCPU          |    0    |  (8, 8, 256)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  17   |       MaxPool2DCPU       |    0    |  (4, 4, 256)  |      (2, 2)       |padd=(0,0), stride=(2,2), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  18   |        Conv2DCPU         | 1180160 |  (4, 4, 512)  | (256, 3, 3, 512)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  19   |         ReluCPU          |    0    |  (4, 4, 512)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  20   |        Conv2DCPU         | 2359808 |  (4, 4, 512)  | (512, 3, 3, 512)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  21   |         ReluCPU          |    0    |  (4, 4, 512)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  22   |        Conv2DCPU         | 2359808 |  (4, 4, 512)  | (512, 3, 3, 512)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  23   |         ReluCPU          |    0    |  (4, 4, 512)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  24   |       MaxPool2DCPU       |    0    |  (2, 2, 512)  |      (2, 2)       |padd=(0,0), stride=(2,2), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  25   |        Conv2DCPU         | 2359808 |  (2, 2, 512)  | (512, 3, 3, 512)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  26   |         ReluCPU          |    0    |  (2, 2, 512)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  27   |        Conv2DCPU         | 2359808 |  (2, 2, 512)  | (512, 3, 3, 512)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  28   |         ReluCPU          |    0    |  (2, 2, 512)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  29   |        Conv2DCPU         | 2359808 |  (2, 2, 512)  | (512, 3, 3, 512)  |padd=(1,1), stride=(1,1), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  30   |         ReluCPU          |    0    |  (2, 2, 512)  |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  31   |       MaxPool2DCPU       |    0    |  (1, 1, 512)  |      (2, 2)       |padd=(0,0), stride=(2,2), dilat=(1,1)|
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  32   |        FlattenCPU        |    0    |    (512,)     |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  33   |          FCCPU           | 262656  |    (512,)     |    (512, 512)     |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  34   |         ReluCPU          |    0    |    (512,)     |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  35   |        DropoutCPU        |    0    |    (512,)     |                   |              rate=0.50              |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  36   |          FCCPU           | 262656  |    (512,)     |    (512, 512)     |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  37   |         ReluCPU          |    0    |    (512,)     |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  38   |        DropoutCPU        |    0    |    (512,)     |                   |              rate=0.50              |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  39   |          FCCPU           |  5130   |     (10,)     |     (512, 10)     |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|  40   |        SoftmaxCPU        |    0    |     (10,)     |                   |                                     |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
|             Total parameters      15245130   58.16 MBytes                                                            |
+-------+--------------------------+---------+---------------+-------------------+-------------------------------------+
**** Loading cifar10 dataset...
**** Parameters:
  model_name                     : vgg16_cifar10
  batch_size                     : 64
  global_batch_size              : None
  dtype                          : <class 'numpy.float32'>
  num_epochs                     : 400
  steps_per_epoch                : 0
  evaluate_on_train              : True
  evaluate_only                  : True
  weights_and_bias_filename      : vgg16-weights-nhwc.npz
  history_file                   : None
  shared_storage                 : False
  enable_fused_bn_relu           : False
  enable_fused_conv_relu         : False
  enable_fused_conv_bn           : False
  enable_fused_conv_bn_relu      : False
  tensor_format                  : NHWC
  enable_best_of                 : False
  dataset_name                   : cifar10
  use_synthetic_data             : False
  dataset_path                   : datasets/cifar10/cifar-10-binary.tar.gz
  test_as_validation             : True
  augment_flip                    : True
  augment_flip_prob               : 0.5
  augment_crop                    : True
  augment_crop_size               : 16
  augment_crop_prob               : 0.5
  validation_split               : 0.2
  optimizer_name                 : sgd
  learning_rate                  : 0.01
  learning_rate_scaling          : True
  momentum                       : 0.9
  decay                          : 0.0001
  nesterov                       : False
  beta1                          : 0.99
  beta2                          : 0.999
  epsilon                        : 1e-07
  rho                            : 0.9
  loss_func                      : categorical_cross_entropy
  metrics                        : categorical_accuracy
  schedulers_names            : warm_up,reduce_lr_on_plateau,model_checkpoint,early_stopping
  warm_up_epochs                 : 5
  early_stopping_metric          : val_categorical_cross_entropy
  early_stopping_patience        : 20
  reduce_lr_on_plateau_metric    : val_categorical_cross_entropy
  reduce_lr_on_plateau_factor    : 0.1
  reduce_lr_on_plateau_patience  : 15
  reduce_lr_on_plateau_min_lr    : 1e-05
  reduce_lr_every_nepochs_factor : 0.5
  reduce_lr_every_nepochs_nepochs: 50
  reduce_lr_every_nepochs_min_lr : 0.001
  stop_at_loss_metric            : val_categorical_accuracy
  stop_at_loss_threshold         : 70.0
  model_checkpoint_metric        : categorical_accuracy
  model_checkpoint_save_freq     : 2
  enable_conv_gemm               : False
  enable_memory_cache            : True
  enable_conv_winograd           : False
  mpi_processes                  : 1
  threads_per_process            : 4
  parallel                       : sequential
  non_blocking_mpi               : False
  gpus_per_node                  : 2
  enable_gpu                     : False
  enable_gpudirect               : False
  enable_nccl                    : False
  enable_cudnn_auto_conv_alg     : True
  tracing                        : True
  tracer_output                  : prueba.trc
  profile                        : False
**** Evaluating on test dataset...
Testing: 100%|██████████████████████| 10000/10000 [00:13<00:00, 715.46 samples/s, test_cce: 0.4376189, test_acc: 89.24%]
```

## Credits
The main contributors, in alphabetically order, to PyDTNN are:
- Adrián Castelló Gimeno ([adcastel@uji.es](mailto:adcastel@uji.es))
- Andrés Enrique Tomás Domínguez ([antodo@upv.es](mailto:antodo@upv.es))
- Enrique Salvador Quintana Ortí ([quintana@uji.es](mailto:quintana@uji.es))
- Jose Ignacio Mestre Miravet ([jmiravet@uji.es](mailto:jmiravet@uji.es))
- Manuel Francisco Dolz Zaragozá ([dolzm@uji.es](mailto:dolzm@uji.es))
- Mar Catalán Carbó ([catalama@uji.es](mailto:catalama@uji.es))
- Miguel Pardo Navarro ([mipardo@uji.es](mailto:mipardo@uji.es))
- Miguel Ángel Prosper Quirós ([mprosper@uji.es](mailto:mprosper@uji.es))
- Paul Ximo Pluijter Izquierdo ([pluijter@uji.es](mailto:pluijter@uji.es))
- Sergio Barrachina Mir ([barrachi@uji.es](mailto:barrachi@uji.es))

If you have questions or comments about PyDTNN, please contact:
- Manuel Francisco Dolz Zaragozá ([dolzm@uji.es](mailto:dolzm@uji.es))

## Citing PyDTNN
If you use PyDTNN, and you would like to acknowledge the project in your
academic publication, we suggest citing the following paper:
- **PyDTNN: A user-friendly and extensible framework for distributed
  deep learning**. Sergio Barrachina, Adrián Castelló, Mar Catalán,
  Manuel F. Dolz, Jose I. Mestre. *Journal of Supercomputing* 77(9), pp.
  9971-9987 (2021) ISSN: 1573-0484. DOI:
  [10.1007/s11227-021-03673-z](http://dx.doi.org/10.1007/s11227-021-03673-z).

Other references:
- **A Flexible Research-Oriented Framework for Distributed Training of
  Deep Neural Networks**. Sergio Barrachina, Adrián Castelló, Mar
  Catalán, Manuel F. Dolz and Jose I. Mestre. *2021 IEEE International
  Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, pp.
  730-739 (2021) DOI:
  [10.1109/IPDPSW52791.2021.00110](http://dx.doi.org/10.1109/IPDPSW52791.2021.00110).

## Acknowledgments
The PyDTNN library has been partially supported by:
- Project TIN2017-82972-R **"Algorithmic Techniques for Energy-Aware and
  Error-Resilient High Performance Computing"** funded by the Spanish
  Ministry of Economy and Competitiveness (2018-2020).
- Project RTI2018-098156-B-C51 **"Innovative Technologies of Processors,
  Accelerators and Networks for Data Centers and High Performance
  Computing"** funded by the Spanish Ministry of Science, Innovation and
  Universities.
- Project CDEIGENT/2017/04 **"High Performance Computing for Neural
  Networks"** funded by the Valencian Government.
- Project UJI-A2019-11 **"Energy-Aware High Performance Computing for
  Deep Neural Networks"** funded by the Universitat Jaume I.
- Project CIDEXG/2022/13 **"AT4SUSDL: Advanced Techniques for
  Sustainable Deep Learning"** funded by the Valencian Government.
- Project RYC2021-033973-I **"Dotación ayuda Ramón y Cajal"** funded by
  the Spanish Ministry of Science, Innovation and Universities.
- Project PID2023-146569NB-C22 **"Inteligencia sostenible en el
  Borde-UJI"** funded by the Spanish Ministry of Science, Innovation and
  Universities.
- Project C121/23 Convenio **"CIBERseguridad post-Cuántica para el
  Aprendizaje Federado en procesadores de bajo consumo y aceleradores
  (CIBER-CAFE)"** funded by the Spanish National Cybersecurity
  Institute.
