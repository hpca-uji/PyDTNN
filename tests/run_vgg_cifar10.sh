# export PYTHONPATH=/home/dolzm/install/extrae-3.6.0/libexec:$PYTHONPATH
# export EXTRAE_CONFIG_FILE=./extrae.xml
# export EXTRAE_ON=1
#
# EXTRAELIB=/home/dolzm/install/extrae-3.6.0/lib/libompitrace.so

NUMNODES=1
NUMPROCS=1
PROCS_PER_NODE=$(($NUMPROCS / $NUMNODES))
export OMP_NUM_THREADS=4

NODETYPE=hexa
LASTH=`echo $NUMNODES - 1 | bc`
HOSTS=$(for i in `seq 0 $LASTH`; do printf "%s%02d," ${NODETYPE} ${i}; done)

# -genv LD_PRELOAD $EXTRAELIB
#mpirun -iface ib0 -hosts $HOSTS -ppn $PROCS_PER_NODE -np $NUMPROCS \
   python3 -u benchmarks_CNN.py \
         --model=vgg3dobn \
         --dataset=cifar10 \
         --dataset_train_path=/Users/mdolz/Downloads/cifar-10-batches-bin/ \
         --dataset_test_path=/Users/mdolz/Downloads/cifar-10-batches-bin/ \
         --test_as_validation=True \
         --batch_size=64 \
         --validation_split=0.2 \
         --steps_per_epoch=0 \
         --num_epochs=100 \
         --evaluate=False \
         --optimizer=sgd \
         --learning_rate=0.001 \
         --momentum=0.9 \
         --decay=0 \
         --loss_func=categorical_accuracy,categorical_cross_entropy \
         --lr_schedulers="" \
         --warm_up_batches=500 \
         --early_stopping_metric=val_categorical_cross_entropy \
         --early_stopping_patience=20 \
         --reduce_lr_on_plateau_metric=val_categorical_cross_entropy \
         --reduce_lr_on_plateau_factor=0.1 \
         --reduce_lr_on_plateau_patience=5 \
         --reduce_lr_on_plateau_min_lr=0 \
         --parallel=sequential \
         --non_blocking_mpi=False \
         --tracing=False \
         --profile=True \
         --enable_gpu=False \
         --dtype=float32
