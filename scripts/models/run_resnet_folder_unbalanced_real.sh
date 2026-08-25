#!/bin/bash

percentage=0.10

case $PMI_RANK in 
  0)
  percentage=0.04
  ;;
  1)
  percentage=0.005
  ;;
  2)
  percentage=0.003
  ;;
  3)
  percentage=0.002
  ;;
esac

echo "PMI_RANK: $PMI_RANK | percentage: $percentage"

pydtnn-benchmark \
  --model=resnet50 \
  --dataset=synthetic \
  --dataset-path=datasets/mnist \
  --test-as-validation=False \
  --batch-size=2 \
  --validation-split=0.2 \
  --steps-per-epoch=0 \
  --num-epochs=5 \
  --evaluate=False \
  --optimizer=sgd \
  --optimizer-nesterov=True \
  --learning-rate=0.1 \
  --optimizer-momentum=0.9 \
  --loss-func=negative_likelihood \
  --metrics=categorical_accuracy,categorical_hinge,categorical_mse,categorical_mae,regression_mse,regression_mae,binary_confusion_matrix,precision,recall,f1_score,multiclass_confusion_matrix \
  --schedulers=warm_up,reduce_lr_on_plateau,early_stopping \
  --warm-up-epochs=5 \
  --early-stopping-metric=val_negative_likelihood \
  --early-stopping-patience=13 \
  --reduce-lr-on-plateau-metric=val_negative_likelihood \
  --reduce-lr-on-plateau-factor=0.5 \
  --reduce-lr-on-plateau-patience=9 \
  --reduce-lr-on-plateau-min-lr=0.00001 \
  --reduce-lr-every-nepochs-nepochs=30 \
  --reduce-lr-every-nepochs-min-lr=0.00001 \
  --reduce-lr-every-nepochs-factor=0.1 \
  --stop-at-loss-metric=val_categorical_accuracy \
  --stop-at-loss-threshold=70.0 \
  --parallel-data=True \
  --shared-data=False \
  --use-blocking-mpi=False \
  --tracing=False \
  --profile=False \
  --backend=cpu \
  --enable-cudnn=True \
  --enable-gpudirect=False \
  --dtype=float32 \
  --tensor-format=nhwc \
  --augment-scale=True \
  --augment-scale-size=250 \
  --shared-tmp-memory=True \
  --model-sync-freq=4 \
  --dataset-percentage=$percentage
