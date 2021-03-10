set -x

# set -x # Debugging flag
export PYTHONPATH=/mnt/beegfs/users/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml
export EXTRAE_ON=1

EXTRAELIB=/mnt/beegfs/users/dolzm/install/extrae-3.6.0/lib/libmpitrace.so

altecnodes=(2 3 4 5 7 8 10)

procs=6
thrds=12

hosts=`for ((i=0;i<procs;i++)); do printf altec%d, ${altecnodes[$i]}; done`

export OMP_NUM_THREADS=$thrds

mpirun -iface ib0 -genv LD_PRELOAD $EXTRAELIB -ppn 1 -np $procs -host $hosts \
   python3 -u benchmarks_CNN.py \
         --model=vgg11bn_cifar10 \
         --dataset=cifar10 \
         --dataset_train_path=/mnt/beegfs/users/dolzm/datasets/cifar-10-batches-bin \
         --dataset_test_path=/mnt/beegfs/users/dolzm/datasets/cifar-10-batches-bin \
         --test_as_validation=True \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=10 \
         --num_epochs=1 \
         --evaluate=False \
         --optimizer=sgd \
         --learning_rate=0.001 \
         --momentum=0.9 \
         --decay=0 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --lr_schedulers=warm_up,stop_at_loss \
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
         --non_blocking_mpi=True \
         --tracing=True \
         --profile=False \
         --enable_gpu=False \
         --dtype=float32 
