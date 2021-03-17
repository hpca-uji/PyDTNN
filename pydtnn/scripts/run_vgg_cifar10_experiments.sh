set -x

altecnodes=(2 3 4 5 7 8 10)
#for procs in 1 2 4 8; do
#for thrds in 24 18 12 6 4 2; do
for procs in 6 4 2 1; do
  for thrds in 24 12 6 2; do

    hosts=$(for ((i = 0; i < procs; i++)); do printf altec%d, ${altecnodes[$i]}; done)

    export OMP_NUM_THREADS=$thrds
    export PYTHONOPTIMIZE=2
    export PYTHONUNBUFFERED="True"

    mpirun -iface ib0 -ppn 1 -np $procs -host $hosts \
      pydtnn_benchmark \
      --model=vgg11bn_cifar10 \
      --dataset=cifar10 \
      --dataset_train_path=/mnt/beegfs/users/dolzm/datasets/cifar-10-batches-bin \
      --dataset_test_path=/mnt/beegfs/users/dolzm/datasets/cifar-10-batches-bin \
      --test_as_validation=True \
      --batch_size=64 \
      --validation_split=0.2 \
      --steps_per_epoch=0 \
      --num_epochs=200 \
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
      --non_blocking_mpi=False \
      --tracing=False \
      --profile=False \
      --enable_gpu=False \
      --history_file="results/result_vgg9_${procs}p_${thrds}t.history" \
      --dtype=float32 | tee results/result_vgg9_${procs}p_${thrds}t.dat
  done
done
