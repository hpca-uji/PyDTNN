set -x

# set -x # Debugging flag
export PYTHONPATH="$HOME/extrae-3.6.0/libexec:$PYTHONPATH"
export EXTRAE_CONFIG_FILE=./extrae.xml
export EXTRAE_ON=1

EXTRAELIB="$HOME/extrae-3.6.0/lib/libmpitrace.so"

altecnodes=(2 3 4 5 7 8 10)

procs=6
thrds=12

hosts=$(for ((i = 0; i < procs; i++)); do printf altec%d, ${altecnodes[$i]}; done)

export OMP_NUM_THREADS=$thrds
export PYTHONOPTIMIZE=2
export PYTHONUNBUFFERED="True"

mpirun -iface ib0 -genv LD_PRELOAD $EXTRAELIB -ppn 1 -np $procs -host $hosts \
  pydtnn_benchmark \
  --model=vgg11bn_cifar10 \
  --dataset=cifar10 \
  --dataset_path=datasets/cifar10/cifar-10-binary.tar.gz \
  --normalize=True \
  --normalize_offset=-0.472 \
  --normalize_scale=1 \
  --test_as_validation=True \
  --batch_size=64 \
  --validation_split=0.2 \
  --steps_per_epoch=10 \
  --num_epochs=50 \
  --evaluate=False \
  --optimizer=sgd \
  --learning_rate=0.001 \
  --momentum=0.9 \
  --decay=0 \
  --loss_func=categorical_accuracy,categorical_cross_entropy \
  --schedulers=warm_up,stop_at_loss \
  --warm_up_epochs=5 \
  --early_stopping_metric=val_categorical_cross_entropy \
  --early_stopping_patience=20 \
  --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
  --reduce_lr_on_plateau_factor=0.1 \
  --reduce_lr_on_plateau_patience=5 \
  --reduce_lr_on_plateau_min_lr=0 \
  --stop_at_loss_metric=val_categorical_accuracy \
  --stop_at_loss_threshold=70.0 \
  --parallel=data \
  --use_blocking_mpi=False \
  --tracing=True \
  --profile=False \
  --enable_gpu=False \
  --dtype=float32
