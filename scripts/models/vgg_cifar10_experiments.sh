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

    MPI_ARGS=()
    export MPICH_UNBUFFERED_STDIO="true"
    if mpirun --version | grep -q 'Open MPI) [5-9].'; then
      MPI_ARGS+=("--output=:raw")
    fi

    mpirun -iface ib0 -ppn 1 -np $procs -host $hosts "${MPI_ARGS[@]}" \
      pydtnn-benchmark \
      --model=vgg11bn_cifar10 \
      --dataset=cifar10 \
      --dataset-path=datasets/cifar10 \
      --input-normalize \
      --test-as-validation \
      --batch-size=64 \
      --validation-split=0.2 \
      --steps-per-epoch=0 \
      --num-epochs=200 \
      --no-evaluate \
      --optimizer=sgd \
      --learning-rate=0.001 \
      --optimizer-momentum=0.9 \
      --optimizer-decay=0 \
      --loss-func=categorical_accuracy,negative_log_likelihood \
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
      --parallel-data \
      --use-blocking-mpi \
      --no-tracing \
      --no-profile \
      --backend=cpu \
      --no-use-cudnn \
      --history-file="results/result_vgg9_${procs}p_${thrds}t.history" \
      --dtype=float32 | tee results/result_vgg9_${procs}p_${thrds}t.dat
  done
done
