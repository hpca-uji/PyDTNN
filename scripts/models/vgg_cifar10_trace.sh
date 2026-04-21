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
  pydtnn-benchmark \
  --model=vgg11bn_cifar10 \
  --dataset=cifar10 \
  --dataset-path=datasets/cifar10 \
  --normalize=True \
  --normalize-offset=-0.472 \
  --normalize-scale=1 \
  --test-as-validation=True \
  --batch-size=64 \
  --validation-split=0.2 \
  --steps-per-epoch=10 \
  --num-epochs=50 \
  --evaluate=False \
  --optimizer=sgd \
  --learning-rate=0.001 \
  --optimizer-momentum=0.9 \
  --optimizer-decay=0 \
  --loss-func=categorical_accuracy,categorical_cross_entropy \
  --schedulers=warm_up,stop_at_loss \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_categorical_cross_entropy \
  --early-stopping-patience=20 \
  --reduce-lr-on-plateau-metric=val_categorical_cross_entropy \
  --reduce-lr-on-plateau-factor=0.1 \
  --reduce-lr-on-plateau-patience=5 \
  --reduce-lr-on-plateau-min-lr=0 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel-data=True \
  --use-blocking-mpi=False \
  --tracing=True \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=False \
  --dtype=float32
