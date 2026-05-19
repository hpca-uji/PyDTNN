# Python Distributed Training of Neural Networks
![](header.svg)

## Introduction
PyDTNN is a lightweight library developed at Universitat Jaume I (Spain)
for distributed and federated deep learning training and inference of
convolutional and transformer-based neural networks, intended as an initial
starting point for interacting with training and inference processes.
PyDTNN prioritizes simplicity over peak performance, offering an approachable
user interface that enables a gentle learning curve. To carry out training
and inference, PyDTNN exploits inter-process parallelism (via MPI) and
intra-process parallelism (via multithreading), leveraging the capabilities
of multicore processors and GPUs at the node level. For this purpose,
PyDTNN relies on mpi4py/pympi/NCCL for message passing, BLAS routines
accessed through NumPy/Cython for multicore processors,
and PyCUDA/CuPy/cuDNN/cuBLAS for GPU acceleration.

Supported layers:
- Fully-connected
- Convolutional 2D
- Max pooling 2D
- Average pooling 2D
- Dropout
- Flatten
- Feed Forward
- Multi-head attention
- Batch normalization
- Encoder & Decoder (for transformer nets, e.g., Bert)
- Addition block (for residual nets, e.g., ResNet)
- Concatenation block (for channel concatenation-based nets, e.g.,
  Inception, GoogleNet, DenseNet, etc.)

Supported datasets:
- **MNIST**: handwritten digit database. This dataset is included into
  the repository. Its binary version can be
  downloaded from: <https://github.com/hpca-uji/PyDTNN>
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
- **IWSLT**: the IWSLT 2017 Multilingual Task addresses text translation,
  including zero-shot translation, with a single MT system across
  all directions including English, German, Dutch, Italian and
  Romanian. As unofficial task, conventional bilingual text
  translation is offered between English and Arabic, French,
  Japanese, Chinese, German and Korean. This dataset is included into
  the repository. Its plain version can be
  downloaded from: <https://github.com/hpca-uji/PyDTNN>
- And others via generic data loaders.

## Installing PyDTNN
```sh
pip install pydtnn
```

Optionally, if you are going to use MPI, you should have installed the
corresponding system libraries, and install the required Python packages
with:
```sh
pip install pydtnn[mpi]
```

Optionally, if you are going to use GPU, you should have installed the
corresponding system libraries, and install the required Python packages
with:
```sh
pip install pydtnn[gpu]
```

Optionally, if you are going to use FHE, you should have installed the
corresponding system libraries, and install the required Python packages
with:
```sh
pip install pydtnn[fhe]
```

Optionally, if you are going to use PyMPI, you can switch protocols
with:
```sh
export PYMPI_PROTO=tcp
export PYMPI_PROTO=grpc
export PYMPI_PROTO=mqtt
```

Optionally, if you are going to use PyMPI with SSL, you should enable it
with:
```sh
export PYMPI_SSL=yes
export PYMPI_SSL_KEY=key.pem    # server private key
export PYMPI_SSL_CERT=cert.pem  # server certificate
```

Optionally, if you are going to use CuPy, you can switch NumPy implementation
with:
```sh
export PYDTNN_CUPY=yes
```

Optionally, if you are going to use CuPy with ROCm, you can switch CuPy implementation
with:
```sh
pip install cupy-rocm-7-0
```

### Contributing and installing from source
Download PyDTNN source code from its GitHub repository and
install it in editable mode:
```sh
git clone https://github.com/hpca-uji/PyDTNN.git
cd PyDTNN
pip install --config-settings editable_mode=compat -e .
```

For more information on how to work on the project see `CONTRIBUTING.md`.

For more information on how to manage external dependencies see `vendor/README.md`.

## Launcher options
The PyDTNN framework comes with a utility launcher called
`pydtnn-benchmark` that supports the following options:

- Model parameters:
  - `--model`: Neural network model: `simplemlp`, `simplecnn`,
    `alexnet`, `vgg11`, `vgg16`, etc. Default: `None`.
  - `--backend`: Backend selection priority.
    Format: `[module[,module[,...]]:]backend[,backend[,...]][;...]`.
    Example: `"all:numpy;conv_2d:gemm;layers,optimizers:numpy,cython"`.
    Selection: More specific modules are attempted first, backend order goes from least to most priority.
    (Note: remember to put value between quotes, specially if there is a ";" in it).
    Default: `cpu`.
  - `--batch-size`: Batch size per MPI rank. Default: `None`.
  - `--global-batch-size`: Batch size between all MPI ranks. Default: `None`.
  - `--dtype`: Datatype to use: `float32`, `float64`. Default: `float32`.
  - `--quantize`: Enable model quantization. Default: `False`.
  - `--quantize-dtype`: Datatype to use: `float32`, `float64`. Default: `float16`.
  - `--num-epochs`: Number of epochs to perform. Default: `1`.
  - `--steps-per-epoch`: Trims the training data depending on the given
    number of steps per epoch. Default: `0`, i.e., do not trim.
  - `--evaluate`: Evaluate the model before and after training the
    model. Default: `False`.
  - `--evaluate-only`: Only evaluate the model. Default: `False`.
  - `--model-state-filename`: Load weights and bias from file.
    Default: `None`.
  - `--history-file`: Filename to save training loss and metrics.
  - `--tensor-format`: Data format to be used: `NHWC` or `NCHW`.
    Optionally, the `AUTO` value sets `NCHW` when cuDNN is available,
    `NHWC` otherwise. Default: `NHWC`.
  - `--random-seed`: Initial state of random number generator. Default: `57005`.
  - `--shared-tmp-memory`: Allows to use a common memory pool for all the temporary data structures.
  - `--shared-data`: If `True` ranks assume they share the file
    system. Default: `True`.
  - `--model-sync-freq`: Number of batches between model synchronization.
    The `0` value synchronizes gradients every batch. Positive values
    synchronizes gradients and weights every N batches. Negative values
    disables synchronization. Default: `0`.
  - `--model-sync-algo`: Aggregation method used to synchronize models:
    `avg`, `wavg` or `invwavg`. Default: `avg`.
  - `--model-sync-participation`: Rank participation to synchronize
    models: `all` or `avail2all`. Default: `all`.
  - `--model-sync-min-avail`: Minimum ranks with data required to
    synchronize models. Default: `0`.
  - `--initial-model-sync`: Synchronize models on training start. Default: `True`.
  - `--final-model-sync`: Synchronize models on training end. Default: `True`.
  - `--model-sync-quantize`: Enable model quantization on synchronize. Default: `False`.
  - `--model-sync-dtype`: Model synchronization quantization target dtype. Default: `float16`.
- Dataset parameters:
  - `--dataset`: Dataset to train: `mnist`, `cifar10`, `synthetic`,
    …. Default: `None`.
  - `--dataset-path`: Path to dataset folder.
  - `--dataset-lang`: Dataset language. Default: `en`.
  - `--dataset-lang2`: Dataset second language. Default: `de`.
  - `--synthetic-train-samples`: Number of synthetic train sample.
    Default: `1000`.
  - `--synthetic-test-samples`: Number of synthetic train sample.
    Default: `100`.
  - `--synthetic-input-shape`: Synthetic input shape (coma separated).
    Default: `3,32,32`.
  - `--synthetic-output-shape`: Synthetic output shape (coma separated).
    Default: `10`.
  - `--dataset-percentage`: Percentage of dataset that will be used. If
    it is `0`: it is deactivated; if is is a value below `1` (and above
    `0`): it will perform undersampling; and if is is a value above `1`:
    it will perform oversampling. Default: `0`.
  - `--test-as-validation`: Prevent making partitions on training data
    for training+validation data, use test data for validation. `True`
    if specified.
  - `--validation-split`: Split between training and validation data.
  - `--augment-flip`: Probability to flip training images. If the value is less or equal to 0 it is disabled. Default: `0.0`.
  - `--augment-rotate`: Probability to rotation training images. If the value is less or equal to 0 it is disabled. Default: `0.0`.
  - `--augment-mask`: Probability to mask training images. If the value is less or equal to 0 it is disabled. Default: `0.0`.
  - `--augment-mask-size`: Size to mask training images. Default: `16`.
  - `--augment-blur`: Probability to blur training images. If the value is less or equal to 0 it is disabled. Default: `0.0`.
  - `--augment-blur-size`: Size to blur training images. Default: `16`.
  - `--validation-split`: Split between training and validation data.
  - `--transform-crop`: Crop the images. `True` if specified.
  - `--transform-crop-perc`: Central crop of the images. Default: `0.875`.
  - `--transform-resize`: Resize the images. `True` if specified.
  - `--transform-resize-size`: New size of the images. Default: `300`.
  - `--normalize`: Normalize dataset. Default: `False`.
  - `--normalize-offset`: Offset samples by a value. Default: `-0.45`.
  - `--normalize-scale`: Scale samples by a value. Default: `3.75`.
- Optimization parameters:
  - `--enable-best-of`: Enable the `BestOf` auto-tuner.
  - `--enable-memory-cache`: Enable the memory cache module to use
    persistent memory.
  - `--enable-fused-bn-relu`: Fuse `BatchNormalization` and `Relu` layers. `True` if specified.
  - `--enable-fused-conv-relu`: Fuse `Conv2D` and `Relu` layers. `True` if specified.
  - `--enable-fused-conv-bn`: Fuse `Conv2D` and `BatchNormalization` layers. `True` if specified.
  - `--enable-fused-conv-bn-relu`: Fuse `Conv2D` and
    `BatchNormalization` and `Relu` layers. Default: `False`.
- Convolution operation parameters:
  - `--conv-direct-method`: ConvDirect algorithm to use in Conv2D layers.
    Default: `convdirect_original_{tensor_format}_default`.
- Optimizer parameters:
  - `--optimizer`: Optimizers: `sgd`, `rmsprop`, `adam`, `nadam`.
    Default: `sgd`.
  - `--learning-rate`: Learning rate. Default: `0.01`.
  - `--learning-rate-scaling`: Scale learning rate in data parallelism:
    `new_lr = lr/num_procs`. `True` if specified.
  - `--optimizer-momentum`: Decay rate for `sgd` optimizer. Default: `0.9`.
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
    batch: `categorical_cross_entropy`, `binary_cross_entropy` or `kl_divergence`.
    Default: `categorical_cross_entropy`.
  - `--metrics`: List of comma-separated metrics that are evaluated on
    each trained batch: `categorical_accuracy`, `categorical_hinge`,
    `categorical_mse`, `categorical_mae`, `regression_mse`,
    `regression_mae`, `binary_confusion_matrix`,
    `multiclass_confusion_matrix`, `precision`, `recall`, `f1_score`.
    Default: `categorical_accuracy`.
- Schedulers parameters:
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
  - `--early-stopping-minimize`: Whether to minimize the metric. If False,
    it will maximize. Default: `True`.
  - `--reduce-lr-on-plateau-metric`: Loss metric monitored by
    `reduce_lr_on_plateau` scheduler. Default:
    `val_categorical_cross_entropy`.
  - `--reduce-lr-on-plateau-factor`: Factor by which the learning rate will be reduced.
    `new_lr = lr *factor`. Default: `0.1`.
  - `--reduce-lr-on-plateau-patience`: Number of epochs with no improvement
    after which LR will be reduced. Default: `5`.
  - `--reduce-lr-on-plateau-min-lr`: Lower bound on the learning rate.
    Default: `0`.
  - `--reduce-lr-every-nepochs-factor`: Factor by which the learning rate
    will be reduced. `new_lr = lr*factor`. Default: `0.1`.
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
    model weights and bias will be saved by the `model_checkpoint` scheduler.
    Default: `2`.
- Parallelization and other performance-related parameters:
  - `--parallel-data`: Enable data parallelization. Default: `False`.
  - `--parallel-pipeline`: Enable pipeline parallelization. Default: `False`.
  - `--use-blocking-mpi`: Enable blocking MPI primitives. Default: `True`.
  - `--use-mpi-buffers`: Enable the use of MPI buffers. Possible values:
    `True` (MPI operations by buffer), `False` (MPI operations by
    object) or `None` (auto-select the better option). Default: `None`.
  - `--enable-gpudirect`: Enable GPU pinned memory for gradients when
    using a CUDA-aware MPI version. Default: `False`.
  - `--enable-nccl`: Enable the use of the `NCCL` library for collective
    communications on GPUs. This option can only be set when cuDNN is available.
    Default: `False`.
  - `--enable-cudnn-auto-conv-algo`: Let `cuDNN` to select the best
    performing convolution algorithm. Default: `True`.
- Encryption parameters:
  - `--encryption`: Encryption library: `tenseal`, `openfhe`, `None`. Default `None`.
  - `--encryption-slots`: Encryption slot count. `2 ^ value`. Default: `12`.
  - `--encryption-scale`: Encryption operational scale. `2 ^ value`. Default: `40`.
  - `--encryption-security`: Encryption security level: `128`, `192`, `256`. Default: `128`.
- Tracing and profiling parameters:
  - `--tracing`: Obtain Simple/Extrae-based traces. Default: `False`.
  - `--tracer-output`: Output file to store the Simple/Extrae-based traces.
  - `--tracer-pmlib-server`: Address of PMlib tracer server. Default: `127.0.0.1`.
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
      --parallel-data=False \
      --tracing=False \
      --profile=False \
      --enable-cudnn=True \
      --backend=gpu \
      --dtype=float32

options
  model                    : simplecnn
  backend                  : gpu
  batch-size               : 64
  global-batch-size        : None
  dtype                    : <class 'numpy.float32'>
  quantize                 : False
  quantize-dtype           : <class 'numpy.float16'>
  num-epochs               : 50
  steps-per-epoch          : 0.0
  evaluate                 : False
  evaluate-only            : False
  model-state-filename: None
  history-file             : 
  tensor-format            : 
  random-seed              : 57005
  shared-tmp-memory        : False

Synchronization options
  shared-data             : True
  model-sync-freq         : 0
  model-sync-algo         : avg
  model-sync-participation: all
  model-sync-min-avail    : 0
  initial-model-sync      : True
  final-model-sync        : True
  model-sync-quantize     : False
  model-sync-dtype        : <class 'numpy.float16'>

Dataset options
  dataset                : mnist
  dataset-percentage     : 0.0
  dataset-path           : datasets/mnist
  dataset-lang           : en
  dataset-lang2          : de
  synthetic-train-samples: 1000
  synthetic-test-samples : 100
  synthetic-input-shape  : 3,32,32
  synthetic-output-shape : 10
  test-as-validation     : False
  validation-split       : 0.2
  augment-shuffle        : True
  augment-flip           : 0.0
  augment-crop           : 0.0
  augment-crop-size      : 16
  transform-crop         : False
  transform-crop-perc    : 0.8
  transform-resize       : False
  transform-resize-size  : 16
  normalize              : False
  normalize-offset       : -0.45
  normalize-scale        : 3.75

Optimization options
  enable-fused-bn-relu     : False
  enable-fused-conv-relu   : False
  enable-fused-conv-bn     : False
  enable-fused-conv-bn-relu: False

Convolution options
  conv-direct-method: 

Optimizer options
  optimizer            : sgd
  learning-rate        : 0.01
  learning-rate-scaling: False
  optimizer-momentum   : 0.9
  optimizer-decay      : 0.0
  optimizer-nesterov   : False
  optimizer-beta1      : 0.99
  optimizer-beta2      : 0.999
  optimizer-epsilon    : 1e-07
  optimizer-rho        : 0.9
  optimizer-tau        : 64
  optimizer-tau-prime  : 32
  optimizer-density    : 0.01
  loss-func            : categorical_cross_entropy
  metrics              : categorical_accuracy

Schedulers options
  schedulers                     : warm_up,reduce_lr_every_nepochs
  warm-up-epochs                 : 5
  early-stopping-metric          : val_categorical_cross_entropy
  early-stopping-patience        : 20
  early-stopping-minimize        : True
  reduce-lr-on-plateau-metric    : val_categorical_cross_entropy
  reduce-lr-on-plateau-factor    : 0.1
  reduce-lr-on-plateau-patience  : 5
  reduce-lr-on-plateau-min-lr    : 0
  reduce-lr-every-nepochs-factor : 0.5
  reduce-lr-every-nepochs-nepochs: 30
  reduce-lr-every-nepochs-min-lr : 0.001
  stop-at-loss-metric            : val_accuracy
  stop-at-loss-threshold         : 0
  model-checkpoint-metric        : val_categorical_cross_entropy
  model-checkpoint-save-freq     : 2

Parallel execution options
  parallel-data             : False
  parallel-pipeline         : False
  use-blocking-mpi          : True
  use-mpi-buffers           : None
  enable-cudnn              : AUTO
  enable-gpudirect          : False
  enable-nccl               : False
  enable-cudnn-auto-conv-alg: True

Encryption options
  encryption         : 
  encryption-slots   : 13
  encryption-scale   : 40
  encryption-security: 128

Tracing options
  tracing            : False
  tracer-output      : 
  tracer-pmlib-server: 127.0.0.1
  tracer-pmlib-port  : 6526
  tracer-pmlib-device: 
  profile            : False
  traceback          : False

Performance modeling options
  cpu-speed   : 4000000000000.0
  memory-bw   : 50000000000.0
  network-bw  : 1000000000.0
  network-lat : 5e-07
  network-algo: vdg

Runtime parallel execution options
  mpi-processes      : 1
  threads-per-process: 16
  gpus-per-node      : 2

Communication options
  mpi-protocol: native
  mpi-server  : 127.0.0.1
  mpi-port    : 61642

Model Summary
=============
- Name: simplecnn
- Dataset: mnist
- Params: 808824
- Memory: 21.19MB
- Optimizer memory: 790.85KB
- Loss memory: 2.75KB
- Metrics memory: 0B
- Input: (1, 28, 28)
- Output: (10,)
- Batch size: 64
- Layers: 12

+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| Id |   Name    | Backend | Memory  | Params |    Input    |   Output    |      Weights      | Padding | Stride | Dilation |  Pool  |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 0  |   Input   | pycuda  | 196.0KB |        |             | (1, 28, 28) |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 1  |  Conv2D   | pycuda  | 3.11MB  |   40   | (1, 28, 28) | (4, 28, 28) |   (4, 1, 3, 3)    | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 2  |   Relu    | pycuda  | 1.53MB  |        | (4, 28, 28) | (4, 28, 28) |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 3  |  Conv2D   | pycuda  |  2.3MB  |  296   | (4, 28, 28) | (8, 28, 28) |   (8, 4, 3, 3)    | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 4  |   Relu    | pycuda  | 3.06MB  |        | (8, 28, 28) | (8, 28, 28) |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 5  | MaxPool2D | pycuda  | 1.91MB  |        | (8, 28, 28) | (8, 14, 14) |                   | (0, 0)  | (2, 2) |  (1, 1)  | (2, 2) |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 6  |  Flatten  | pycuda  |         |        | (8, 14, 14) |   (1568,)   |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 7  |    FC     | pycuda  | 1.95MB  | 803328 |   (1568,)   |   (128,)    | (1568, 128, 1, 1) |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 8  |   Relu    | pycuda  | 64.0KB  |        |   (128,)    |   (128,)    |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 9  |  Dropout  | pycuda  | 6.25MB  |        |   (128,)    |   (128,)    |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 10 |    FC     | pycuda  | 44.83KB |  5160  |   (128,)    |    (10,)    |  (128, 10, 1, 1)  |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+
| 11 |  Softmax  | pycuda  |  5.0KB  |        |    (10,)    |    (10,)    |                   |         |        |          |        |
+----+-----------+---------+---------+--------+-------------+-------------+-------------------+---------+--------+----------+--------+

**** Training...
Epoch  1/50: 100%|████████| 48000/48000 [00:13<00:00, 3455.71 samples/s, train_cce: 1.1184705, train_acc: 62.32%, val_cce: 0.7023183, val_acc: 77.95%]
Epoch  2/50: 100%|████████| 48000/48000 [00:12<00:00, 3752.62 samples/s, train_cce: 0.6748989, train_acc: 78.77%, val_cce: 0.5968420, val_acc: 81.47%]
Epoch  3/50: 100%|████████| 48000/48000 [00:12<00:00, 3809.99 samples/s, train_cce: 0.5856293, train_acc: 81.59%, val_cce: 0.5133920, val_acc: 83.88%]
Epoch  4/50: 100%|████████| 48000/48000 [00:13<00:00, 3598.30 samples/s, train_cce: 0.5017195, train_acc: 84.08%, val_cce: 0.4584806, val_acc: 85.58%]
Epoch  5/50: 100%|████████| 48000/48000 [00:13<00:00, 3666.76 samples/s, train_cce: 0.4397651, train_acc: 86.20%, val_cce: 0.4184231, val_acc: 86.84%]
Epoch  6/50: 100%|████████| 48000/48000 [00:13<00:00, 3453.78 samples/s, train_cce: 0.4097258, train_acc: 86.96%, val_cce: 0.3892847, val_acc: 87.96%]
Epoch  7/50: 100%|████████| 48000/48000 [00:13<00:00, 3458.85 samples/s, train_cce: 0.3888468, train_acc: 87.68%, val_cce: 0.3704451, val_acc: 88.19%]
Epoch  8/50: 100%|████████| 48000/48000 [00:13<00:00, 3662.07 samples/s, train_cce: 0.3821366, train_acc: 87.90%, val_cce: 0.3650805, val_acc: 88.22%]
Epoch  9/50: 100%|████████| 48000/48000 [00:12<00:00, 3802.80 samples/s, train_cce: 0.3780845, train_acc: 88.06%, val_cce: 0.3555509, val_acc: 88.73%]
Epoch 10/50: 100%|████████| 48000/48000 [00:14<00:00, 3404.88 samples/s, train_cce: 0.3715006, train_acc: 87.95%, val_cce: 0.3681258, val_acc: 88.07%]
Epoch 11/50: 100%|████████| 48000/48000 [00:13<00:00, 3579.19 samples/s, train_cce: 0.3639101, train_acc: 88.19%, val_cce: 0.3594698, val_acc: 88.51%]
Epoch 12/50: 100%|████████| 48000/48000 [00:13<00:00, 3546.87 samples/s, train_cce: 0.3627393, train_acc: 88.31%, val_cce: 0.3599054, val_acc: 88.11%]
Epoch 13/50: 100%|████████| 48000/48000 [00:13<00:00, 3634.02 samples/s, train_cce: 0.3599829, train_acc: 88.59%, val_cce: 0.3425802, val_acc: 89.25%]
Epoch 14/50: 100%|████████| 48000/48000 [00:13<00:00, 3542.70 samples/s, train_cce: 0.3640404, train_acc: 88.40%, val_cce: 0.3528128, val_acc: 88.71%]
Epoch 15/50: 100%|████████| 48000/48000 [00:13<00:00, 3505.95 samples/s, train_cce: 0.3599551, train_acc: 88.48%, val_cce: 0.3551110, val_acc: 88.88%]
Epoch 16/50: 100%|████████| 48000/48000 [00:12<00:00, 3711.92 samples/s, train_cce: 0.3572799, train_acc: 88.57%, val_cce: 0.3349612, val_acc: 89.40%]
Epoch 17/50: 100%|████████| 48000/48000 [00:13<00:00, 3575.03 samples/s, train_cce: 0.3551838, train_acc: 88.65%, val_cce: 0.3405531, val_acc: 89.38%]
Epoch 18/50: 100%|████████| 48000/48000 [00:13<00:00, 3624.12 samples/s, train_cce: 0.3534711, train_acc: 88.79%, val_cce: 0.3470441, val_acc: 89.01%]
Epoch 19/50: 100%|████████| 48000/48000 [00:13<00:00, 3442.18 samples/s, train_cce: 0.3484907, train_acc: 88.92%, val_cce: 0.3311701, val_acc: 89.67%]
Epoch 20/50: 100%|████████| 48000/48000 [00:14<00:00, 3423.23 samples/s, train_cce: 0.3441238, train_acc: 88.95%, val_cce: 0.3360619, val_acc: 89.24%]
Epoch 21/50: 100%|████████| 48000/48000 [00:13<00:00, 3587.19 samples/s, train_cce: 0.3479733, train_acc: 88.88%, val_cce: 0.3403825, val_acc: 89.65%]
Epoch 22/50: 100%|████████| 48000/48000 [00:13<00:00, 3546.39 samples/s, train_cce: 0.3463119, train_acc: 88.89%, val_cce: 0.3362577, val_acc: 89.77%]
Epoch 23/50: 100%|████████| 48000/48000 [00:13<00:00, 3506.51 samples/s, train_cce: 0.3460476, train_acc: 88.86%, val_cce: 0.3373249, val_acc: 89.10%]
Epoch 24/50: 100%|████████| 48000/48000 [00:12<00:00, 3708.42 samples/s, train_cce: 0.3435737, train_acc: 89.05%, val_cce: 0.3395897, val_acc: 89.25%]
Epoch 25/50: 100%|████████| 48000/48000 [00:13<00:00, 3525.08 samples/s, train_cce: 0.3438735, train_acc: 89.26%, val_cce: 0.3333109, val_acc: 89.50%]
Epoch 26/50: 100%|████████| 48000/48000 [00:13<00:00, 3566.45 samples/s, train_cce: 0.3414197, train_acc: 89.10%, val_cce: 0.3265650, val_acc: 89.43%]
Epoch 27/50: 100%|████████| 48000/48000 [00:13<00:00, 3579.69 samples/s, train_cce: 0.3437447, train_acc: 89.01%, val_cce: 0.3319236, val_acc: 89.57%]
Epoch 28/50: 100%|████████| 48000/48000 [00:13<00:00, 3590.92 samples/s, train_cce: 0.3387754, train_acc: 89.22%, val_cce: 0.3345976, val_acc: 89.70%]
Epoch 29/50: 100%|████████| 48000/48000 [00:13<00:00, 3481.79 samples/s, train_cce: 0.3417804, train_acc: 89.12%, val_cce: 0.3417696, val_acc: 89.32%]
Epoch 30/50: 100%|███████| 48000/48000 [00:13<00:00, 24522.59 samples/s, train_cce: 0.3385290, train_acc: 89.17%, val_cce: 0.3353500, val_acc: 89.52%]
Scheduler ReduceLREveryNEpochs: Setting learning rate to 0.00500000!
Epoch 30/50: 100%|████████| 48000/48000 [00:13<00:00, 3596.27 samples/s, train_cce: 0.3385290, train_acc: 89.17%, val_cce: 0.3353500, val_acc: 89.52%]
Epoch 31/50: 100%|████████| 48000/48000 [00:13<00:00, 3543.55 samples/s, train_cce: 0.3262158, train_acc: 89.56%, val_cce: 0.3149563, val_acc: 90.25%]
Epoch 32/50: 100%|████████| 48000/48000 [00:13<00:00, 3496.25 samples/s, train_cce: 0.3222284, train_acc: 89.81%, val_cce: 0.3056272, val_acc: 90.32%]
Epoch 33/50: 100%|████████| 48000/48000 [00:12<00:00, 3705.79 samples/s, train_cce: 0.3210438, train_acc: 89.67%, val_cce: 0.3245406, val_acc: 89.61%]
Epoch 34/50: 100%|████████| 48000/48000 [00:12<00:00, 3820.55 samples/s, train_cce: 0.3233700, train_acc: 89.63%, val_cce: 0.3098261, val_acc: 90.35%]
Epoch 35/50: 100%|████████| 48000/48000 [00:13<00:00, 3568.94 samples/s, train_cce: 0.3225272, train_acc: 89.76%, val_cce: 0.3156418, val_acc: 90.00%]
Epoch 36/50: 100%|████████| 48000/48000 [00:13<00:00, 3583.64 samples/s, train_cce: 0.3222091, train_acc: 89.83%, val_cce: 0.3227611, val_acc: 89.84%]
Epoch 37/50: 100%|████████| 48000/48000 [00:13<00:00, 3689.22 samples/s, train_cce: 0.3263476, train_acc: 89.64%, val_cce: 0.3303543, val_acc: 89.53%]
Epoch 38/50: 100%|████████| 48000/48000 [00:12<00:00, 3818.57 samples/s, train_cce: 0.3201116, train_acc: 89.90%, val_cce: 0.3213890, val_acc: 90.01%]
Epoch 39/50: 100%|████████| 48000/48000 [00:13<00:00, 3457.16 samples/s, train_cce: 0.3255080, train_acc: 89.51%, val_cce: 0.3159171, val_acc: 90.02%]
Epoch 40/50: 100%|████████| 48000/48000 [00:13<00:00, 3548.30 samples/s, train_cce: 0.3240053, train_acc: 89.63%, val_cce: 0.3084729, val_acc: 90.22%]
Epoch 41/50: 100%|████████| 48000/48000 [00:13<00:00, 3508.52 samples/s, train_cce: 0.3220426, train_acc: 89.73%, val_cce: 0.3169541, val_acc: 89.98%]
Epoch 42/50: 100%|████████| 48000/48000 [00:13<00:00, 3622.97 samples/s, train_cce: 0.3210119, train_acc: 89.58%, val_cce: 0.3271742, val_acc: 89.85%]
Epoch 43/50: 100%|████████| 48000/48000 [00:12<00:00, 3744.07 samples/s, train_cce: 0.3192258, train_acc: 89.86%, val_cce: 0.3215209, val_acc: 89.89%]
Epoch 44/50: 100%|████████| 48000/48000 [00:12<00:00, 3809.49 samples/s, train_cce: 0.3244888, train_acc: 89.55%, val_cce: 0.3067044, val_acc: 90.12%]
Epoch 45/50: 100%|████████| 48000/48000 [00:13<00:00, 3626.12 samples/s, train_cce: 0.3195171, train_acc: 89.81%, val_cce: 0.3191177, val_acc: 90.17%]
Epoch 46/50: 100%|████████| 48000/48000 [00:13<00:00, 3568.72 samples/s, train_cce: 0.3168551, train_acc: 89.89%, val_cce: 0.3225888, val_acc: 89.83%]
Epoch 47/50: 100%|████████| 48000/48000 [00:13<00:00, 3486.95 samples/s, train_cce: 0.3193468, train_acc: 89.84%, val_cce: 0.3129438, val_acc: 90.23%]
Epoch 48/50: 100%|████████| 48000/48000 [00:12<00:00, 3708.50 samples/s, train_cce: 0.3167030, train_acc: 89.94%, val_cce: 0.3166687, val_acc: 90.21%]
Epoch 49/50: 100%|████████| 48000/48000 [00:12<00:00, 3741.18 samples/s, train_cce: 0.3168798, train_acc: 89.89%, val_cce: 0.3169060, val_acc: 90.00%]
Epoch 50/50: 100%|████████| 48000/48000 [00:12<00:00, 3755.39 samples/s, train_cce: 0.3211099, train_acc: 89.83%, val_cce: 0.3139842, val_acc: 90.19%]

**** Done...
Training and validation time: 691.9629 s
Training and validation time per epoch: 691.9629 s
Training and validation throughput: 69.3679 samples/s

 -------------------------------------
| Performance counter training report |
 -------------------------------------
Training time (from model): 60.6756 s
Training time per epoch (from model): 60.6756 s
Training throughput (from model): 39554.6417 samples/s
Training time (from model, estimated from last half of each epoch): 59.9202 s
Training throughput (from model, from last half of each epoch): 40053.2911 samples/s
Training maximum memory allocated: 7590.83 MiB
Training mean memory allocated: 5222.03 MiB
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
    --model-state-filename=vgg16-weights-nhwc.npz \
    --tracing=False \
    --profile=False \
    --enable-cudnn=True \
    --backend=gpu \
    --dtype=float32

options
  model                    : vgg16_cifar10
  backend                  : cpu
  batch-size               : 64
  global-batch-size        : None
  dtype                    : <class 'numpy.float32'>
  quantize                 : False
  quantize-dtype           : <class 'numpy.float16'>
  num-epochs               : 400
  steps-per-epoch          : 0.0
  evaluate                 : False
  evaluate-only            : True
  weights-and-bias-filename: model-vgg16_cifar10-weights-rank_0-20260506.npz
  history-file             : result_vgg_cifar.yaml
  tensor-format            : 
  random-seed              : 57005
  shared-tmp-memory        : False

Synchronization options
  shared-data             : True
  model-sync-freq         : 0
  model-sync-algo         : avg
  model-sync-participation: all
  model-sync-min-avail    : 0
  initial-model-sync      : True
  final-model-sync        : True
  model-sync-quantize     : False
  model-sync-dtype        : <class 'numpy.float16'>

Dataset options
  dataset                : cifar10
  dataset-percentage     : 0.0
  dataset-path           : datasets/cifar10
  dataset-lang           : en
  dataset-lang2          : de
  synthetic-train-samples: 1000
  synthetic-test-samples : 100
  synthetic-input-shape  : 3,32,32
  synthetic-output-shape : 10
  test-as-validation     : True
  validation-split       : 0.2
  augment-shuffle        : True
  augment-flip           : 0.5
  augment-crop           : 0.5
  augment-crop-size      : 16
  transform-crop         : False
  transform-crop-perc    : 0.875
  transform-resize       : False
  transform-resize-size  : 300
  normalize              : True
  normalize-offset       : -0.472
  normalize-scale        : 1.0

Optimization options
  enable-fused-bn-relu     : False
  enable-fused-conv-relu   : False
  enable-fused-conv-bn     : False
  enable-fused-conv-bn-relu: False

Convolution options
  conv-direct-method: 

Optimizer options
  optimizer            : sgd
  learning-rate        : 0.01
  learning-rate-scaling: False
  optimizer-momentum   : 0.9
  optimizer-decay      : 0.0001
  optimizer-nesterov   : False
  optimizer-beta1      : 0.99
  optimizer-beta2      : 0.999
  optimizer-epsilon    : 1e-07
  optimizer-rho        : 0.9
  optimizer-tau        : 64
  optimizer-tau-prime  : 32
  optimizer-density    : 0.01
  loss-func            : categorical_cross_entropy
  metrics              : categorical_accuracy

Schedulers options
  schedulers                     : warm_up,reduce_lr_on_plateau,early_stopping
  warm-up-epochs                 : 5
  early-stopping-metric          : val_categorical_cross_entropy
  early-stopping-patience        : 20
  early-stopping-minimize        : True
  reduce-lr-on-plateau-metric    : val_categorical_cross_entropy
  reduce-lr-on-plateau-factor    : 0.1
  reduce-lr-on-plateau-patience  : 15
  reduce-lr-on-plateau-min-lr    : 0.0001
  reduce-lr-every-nepochs-factor : 0.5
  reduce-lr-every-nepochs-nepochs: 50
  reduce-lr-every-nepochs-min-lr : 0.001
  stop-at-loss-metric            : val_categorical_accuracy
  stop-at-loss-threshold         : 70.0
  model-checkpoint-metric        : val_categorical_cross_entropy
  model-checkpoint-save-freq     : 2

Parallel execution options
  parallel-data              : False
  parallel-pipeline          : False
  use-blocking-mpi           : True
  use-mpi-buffers            : None
  enable-cudnn               : AUTO
  enable-gpudirect           : False
  enable-nccl                : False
  enable-cudnn-auto-conv-algo: False

Encryption options
  encryption         : 
  encryption-slots   : 13
  encryption-scale   : 40
  encryption-security: 128

Tracing options
  tracing            : False
  tracer-output      : 
  tracer-pmlib-server: 127.0.0.1
  tracer-pmlib-port  : 6526
  tracer-pmlib-device: 
  profile            : False
  traceback          : False

Performance modeling options
  cpu-speed   : 4000000000000.0
  memory-bw   : 50000000000.0
  network-bw  : 1000000000.0
  network-lat : 5e-07
  network-algo: vdg

Runtime parallel execution options
  mpi-processes      : 1
  threads-per-process: 8
  gpus-per-node      : 2

Communication options
  mpi-protocol: native
  mpi-server  : 127.0.0.1
  mpi-port    : 61642

Model Summary
=============
- Name: vgg16_cifar10
- Dataset: cifar10
- Params: 33638218
- Memory: 1.39GB (256.65MB tmp)
- Optimizer memory: 384.96MB (256.64MB tmp)
- Loss memory: 5.5KB (3.0KB tmp)
- Metrics memory: 256.0B (256.0B tmp)
- Input: (3, 32, 32)
- Output: (10,)
- Batch size: 64
- Layers: 41

+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| Id |   Name    | Backend |       Memory        |  Params  |     Input     |    Output     |     Weights      | Padding | Stride | Dilation |  Pool  |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 0  |   Input   |  numpy  |                     |          |               |  (3, 32, 32)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 1  |  Conv2D   | cython  |       22.76MB       |   1792   |  (3, 32, 32)  | (64, 32, 32)  |  (64, 3, 3, 3)   | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 2  |   Relu    | cython  |       20.0MB        |          | (64, 32, 32)  | (64, 32, 32)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 3  |  Conv2D   | cython  |      160.14MB       |  36928   | (64, 32, 32)  | (64, 32, 32)  |  (64, 64, 3, 3)  | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 4  |   Relu    | cython  |       20.0MB        |          | (64, 32, 32)  | (64, 32, 32)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 5  | MaxPool2D | cython  |        8.0MB        |          | (64, 32, 32)  | (64, 16, 16)  |                  | (0, 0)  | (2, 2) |  (1, 1)  | (2, 2) |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 6  |  Conv2D   | cython  |       44.28MB       |  73856   | (64, 16, 16)  | (128, 16, 16) | (128, 64, 3, 3)  | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 7  |   Relu    | cython  |       10.0MB        |          | (128, 16, 16) | (128, 16, 16) |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 8  |  Conv2D   | cython  |       80.56MB       |  147584  | (128, 16, 16) | (128, 16, 16) | (128, 128, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 9  |   Relu    | cython  |       10.0MB        |          | (128, 16, 16) | (128, 16, 16) |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 10 | MaxPool2D | cython  |        4.0MB        |          | (128, 16, 16) |  (128, 8, 8)  |                  | (0, 0)  | (2, 2) |  (1, 1)  | (2, 2) |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 11 |  Conv2D   | cython  |       23.13MB       |  295168  |  (128, 8, 8)  |  (256, 8, 8)  | (256, 128, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 12 |   Relu    | cython  |        5.0MB        |          |  (256, 8, 8)  |  (256, 8, 8)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 13 |  Conv2D   | cython  |       42.25MB       |  590080  |  (256, 8, 8)  |  (256, 8, 8)  | (256, 256, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 14 |   Relu    | cython  |        5.0MB        |          |  (256, 8, 8)  |  (256, 8, 8)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 15 |  Conv2D   | cython  |       42.25MB       |  590080  |  (256, 8, 8)  |  (256, 8, 8)  | (256, 256, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 16 |   Relu    | cython  |        5.0MB        |          |  (256, 8, 8)  |  (256, 8, 8)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 17 | MaxPool2D | cython  |        2.0MB        |          |  (256, 8, 8)  |  (256, 4, 4)  |                  | (0, 0)  | (2, 2) |  (1, 1)  | (2, 2) |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 18 |  Conv2D   | cython  |       15.5MB        | 1180160  |  (256, 4, 4)  |  (512, 4, 4)  | (512, 256, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 19 |   Relu    | cython  |        2.5MB        |          |  (512, 4, 4)  |  (512, 4, 4)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 20 |  Conv2D   | cython  |       29.0MB        | 2359808  |  (512, 4, 4)  |  (512, 4, 4)  | (512, 512, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 21 |   Relu    | cython  |        2.5MB        |          |  (512, 4, 4)  |  (512, 4, 4)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 22 |  Conv2D   | cython  |       29.0MB        | 2359808  |  (512, 4, 4)  |  (512, 4, 4)  | (512, 512, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 23 |   Relu    | cython  |        2.5MB        |          |  (512, 4, 4)  |  (512, 4, 4)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 24 | MaxPool2D | cython  |        1.0MB        |          |  (512, 4, 4)  |  (512, 2, 2)  |                  | (0, 0)  | (2, 2) |  (1, 1)  | (2, 2) |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 25 |  Conv2D   | cython  |       14.0MB        | 2359808  |  (512, 2, 2)  |  (512, 2, 2)  | (512, 512, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 26 |   Relu    | cython  |       640.0KB       |          |  (512, 2, 2)  |  (512, 2, 2)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 27 |  Conv2D   | cython  |       14.0MB        | 2359808  |  (512, 2, 2)  |  (512, 2, 2)  | (512, 512, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 28 |   Relu    | cython  |       640.0KB       |          |  (512, 2, 2)  |  (512, 2, 2)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 29 |  Conv2D   | cython  |       14.0MB        | 2359808  |  (512, 2, 2)  |  (512, 2, 2)  | (512, 512, 3, 3) | (1, 1)  | (1, 1) |  (1, 1)  |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 30 |   Relu    | cython  |       640.0KB       |          |  (512, 2, 2)  |  (512, 2, 2)  |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 31 | MaxPool2D | cython  |       256.0KB       |          |  (512, 2, 2)  |  (512, 1, 1)  |                  | (0, 0)  | (2, 2) |  (1, 1)  | (2, 2) |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 32 |  Flatten  |  numpy  |                     |          |  (512, 1, 1)  |    (512,)     |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 33 |    FC     |  numpy  |       17.14MB       | 2101248  |    (512,)     |    (4096,)    |   (512, 4096)    |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 34 |   Relu    | cython  |       1.25MB        |          |    (4096,)    |    (4096,)    |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 35 |  Dropout  |  numpy  |       16.0KB        |          |    (4096,)    |    (4096,)    |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 36 |    FC     |  numpy  |      130.02MB       | 16781312 |    (4096,)    |    (4096,)    |   (4096, 4096)   |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 37 |   Relu    | cython  |       1.25MB        |          |    (4096,)    |    (4096,)    |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 38 |  Dropout  |  numpy  |       16.0KB        |          |    (4096,)    |    (4096,)    |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 39 |    FC     |  numpy  |       1.31MB        |  40970   |    (4096,)    |     (10,)     |    (4096, 10)    |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+
| 40 |  Softmax  |  numpy  | 5.75KB (3.25KB tmp) |          |     (10,)     |     (10,)     |                  |         |        |          |        |
+----+-----------+---------+---------------------+----------+---------------+---------------+------------------+---------+--------+----------+--------+

**** Evaluating on test dataset...
Testing: 100%|████████████████████████████████████████████████████| 10000/10000 [00:28<00:00, 346.26 samples/s, test_cce: 0.7342598, test_acc: 77.91%]
Testing time: 29.3891 s
Testing throughput: 340.2627 samples/s

 ------------------------------------
| Performance counter testing report |
 ------------------------------------
Testing time (from model): 28.3266 s
Testing throughput (from model): 353.0247 samples/s
Testing maximum memory allocated: 2205.47 MiB
Testing mean memory allocated: 2205.47 MiB
```

## Credits
The main contributors to PyDTNN are:
- Miguel Ángel Prosper Quirós ([mprosper@uji.es](mailto:mprosper@uji.es))
- Paul Ximo Pluijter Izquierdo ([pluijter@uji.es](mailto:pluijter@uji.es))
- Manuel Francisco Dolz Zaragozá ([dolzm@uji.es](mailto:dolzm@uji.es))
- Sergio Barrachina Mir ([barrachi@uji.es](mailto:barrachi@uji.es))
- Miguel Pardo Navarro ([mipardo@uji.es](mailto:mipardo@uji.es))
- Andrés Enrique Tomás Domínguez ([antodo@upv.es](mailto:antodo@upv.es))
- Adrián Castelló Gimeno ([adcastel@uji.es](mailto:adcastel@uji.es))
- Adrián Bartolomé López ([abartolo@uji.es](mailto:abartolo@uji.es))
- Mar Catalán Carbó ([catalama@uji.es](mailto:catalama@uji.es))
- Jose Ignacio Mestre Miravet ([jmiravet@uji.es](mailto:jmiravet@uji.es))
- Enrique Salvador Quintana Ortí ([quintana@uji.es](mailto:quintana@uji.es))

Other colaborators of the project:
- Pau San Juan ([p.sanjuan@upv.es](mailto:p.sanjuan@upv.es))
- Pedro Alonso-Jordá ([palonso@uji.es](mailto:palonso@uji.es))

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

![](footer.jpg)