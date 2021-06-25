"""
PyDTNN parser

The parser in this module will be used by 'pydtnn_benchmark' to parse the
command line arguments.

And what is even more important, it will also be loaded by the Model class to
obtain default values to its non mandatory attributes. This way, when a model
object is instantiated (even if it is not from 'pydtnn_benchmark') it will
initially have default values for all the attributes declared on the parser.

If you want to define a new option, just declare it here. It will automatically
be available as a Model attribute.
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import argparse
import os

import numpy as np


def bool_lambda(x):
    """Returns True if command line value is any of true, 1, or yes"""
    return str(x).lower() in ['true', '1', 'yes']


def np_dtype(x):
    """Returns a numpy object from an string representing the data type"""
    return getattr(np, x)


_this_file_path = os.path.dirname(os.path.realpath(__file__))
_scripts_path = os.path.join(_this_file_path, "scripts")
_default_dataset_path = os.path.join(_this_file_path, "datasets/mnist")
_desc = "Trains or evaluates a neural network using PyDTNN."
_epilogue = f"""Example scripts that call this program for training
and evaluating different neural network models with different datasets are
available at: '{_scripts_path}'."""

# Parser and the supported arguments with their default values
parser = argparse.ArgumentParser(description=_desc, epilog=_epilogue)

# Model
parser.add_argument('--model', dest="model_name", type=str, default="simplecnn")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--global_batch_size', type=int, default=None)
parser.add_argument('--dtype', type=np_dtype, default=np.float32)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--steps_per_epoch', type=int, default=0)
parser.add_argument('--evaluate', dest="evaluate_on_train", default=False, type=bool_lambda)
parser.add_argument('--evaluate_only', default=False, type=bool_lambda)
parser.add_argument('--weights_and_bias_filename', type=str, default=None)
parser.add_argument('--history_file', type=str, default=None)
parser.add_argument('--shared_storage', default=False, type=bool_lambda)
parser.add_argument('--enable_fused_relus', type=bool_lambda, default=False)
parser.add_argument('--tensor_format', type=lambda s: s.upper(), default="NHWC")
parser.add_argument('--enable_best_of', type=bool_lambda, default=False)

# Dataset options
_ds_group = parser.add_argument_group("Dataset options")
_ds_group.add_argument('--dataset', dest="dataset_name", type=str, default="mnist")
_ds_group.add_argument('--use_synthetic_data', default=False, type=bool_lambda)
_ds_group.add_argument('--dataset_train_path', type=str, default=_default_dataset_path)
_ds_group.add_argument('--dataset_test_path', type=str, default=_default_dataset_path)
_ds_group.add_argument('--test_as_validation', default=False, type=bool_lambda)
_ds_group.add_argument('--flip_images', default=False, type=bool_lambda)
_ds_group.add_argument('--flip_images_prob', type=float, default=0.5)
_ds_group.add_argument('--crop_images', default=False, type=bool_lambda)
_ds_group.add_argument('--crop_images_size', type=int, default=16)
_ds_group.add_argument('--crop_images_prob', type=float, default=0.5)
_ds_group.add_argument('--validation_split', type=float, default=0.0)

# Optimizer options
_op_group = parser.add_argument_group("Optimizer options")
_op_group.add_argument('--optimizer', dest="optimizer_name", type=str, default="sgd")
_op_group.add_argument('--learning_rate', type=float, default=1e-2)
_op_group.add_argument('--learning_rate_scaling', default=True, type=bool_lambda)
_op_group.add_argument('--momentum', type=float, default=0.9)
_op_group.add_argument('--decay', type=float, default=0.0)
_op_group.add_argument('--nesterov', default=False, type=bool_lambda)
_op_group.add_argument('--beta1', type=float, default=0.99)
_op_group.add_argument('--beta2', type=float, default=0.999)
_op_group.add_argument('--epsilon', type=float, default=1e-7)
_op_group.add_argument('--rho', type=float, default=0.9)
_op_group.add_argument('--loss_func', type=str, default="categorical_cross_entropy")
_op_group.add_argument('--metrics', type=str, default="categorical_accuracy")

# Learning rate schedulers options
_lr_group = parser.add_argument_group("Learning rate schedulers options")
_lr_group.add_argument('--lr_schedulers', dest="lr_schedulers_names", type=str,
                       default="early_stopping,reduce_lr_on_plateau,model_checkpoint")
_lr_group.add_argument('--warm_up_epochs', type=int, default=5)
_lr_group.add_argument('--early_stopping_metric', type=str, default="val_categorical_cross_entropy")
_lr_group.add_argument('--early_stopping_patience', type=int, default=10)
_lr_group.add_argument('--reduce_lr_on_plateau_metric', type=str, default="val_categorical_cross_entropy")
_lr_group.add_argument('--reduce_lr_on_plateau_factor', type=float, default=0.1)
_lr_group.add_argument('--reduce_lr_on_plateau_patience', type=int, default=5)
_lr_group.add_argument('--reduce_lr_on_plateau_min_lr', type=float, default=0)
_lr_group.add_argument('--reduce_lr_every_nepochs_factor', type=float, default=0.1)
_lr_group.add_argument('--reduce_lr_every_nepochs_nepochs', type=int, default=5)
_lr_group.add_argument('--reduce_lr_every_nepochs_min_lr', type=float, default=0)
_lr_group.add_argument('--stop_at_loss_metric', type=str, default="val_accuracy")
_lr_group.add_argument('--stop_at_loss_threshold', type=float, default=0)
_lr_group.add_argument('--model_checkpoint_metric', type=str, default="val_categorical_cross_entropy")
_lr_group.add_argument('--model_checkpoint_save_freq', type=int, default=2)

# ConvGemm
_cg_group = parser.add_argument_group("ConvGemm options")
_cg_group.add_argument('--enable_conv_gemm', type=bool_lambda, default=False)
_cg_group.add_argument('--conv_gemm_fallback_to_im2col', type=bool_lambda, default=False)
_cg_group.add_argument('--conv_gemm_cache', type=bool_lambda, default=True)
_cg_group.add_argument('--conv_gemm_deconv', type=bool_lambda, default=False)
_cg_group.add_argument('--conv_gemm_trans', type=bool_lambda, default=False)

# Parallel execution options
_pe_group = parser.add_argument_group("Parallel execution options")
_pe_group.add_argument('--mpi_processes', type=int, default=1, help=argparse.SUPPRESS)
_pe_group.add_argument('--threads_per_process', type=int, default=1, help=argparse.SUPPRESS)
_pe_group.add_argument('--parallel', type=str, default="sequential")
_pe_group.add_argument('--non_blocking_mpi', type=bool_lambda, default=False)
_pe_group.add_argument('--gpus_per_node', type=int, default=1, help=argparse.SUPPRESS)
_pe_group.add_argument('--enable_gpu', type=bool_lambda, default=False)
_pe_group.add_argument('--enable_gpudirect', type=bool_lambda, default=False)
_pe_group.add_argument('--enable_nccl', type=bool_lambda, default=False)
_pe_group.add_argument('--enable_cudnn_auto_conv_alg', type=bool_lambda, default=True)

# Tracing and profiling
_tr_group = parser.add_argument_group("Tracing options")
_tr_group.add_argument('--tracing', type=bool_lambda, default=False)
_tr_group.add_argument('--tracer_output', type=str, default="")
_tr_group.add_argument('--profile', type=bool_lambda, default=False)

# Performance modeling options (argparse.SUPPRESS is used to avoid showing them on the message)
_pm_group = parser.add_argument_group("Performance modeling options")
_pm_group.add_argument('--cpu_speed', type=float, default=4e12, help=argparse.SUPPRESS)
_pm_group.add_argument('--memory_bw', type=float, default=50e9, help=argparse.SUPPRESS)
_pm_group.add_argument('--network_bw', type=float, default=1e9, help=argparse.SUPPRESS)
_pm_group.add_argument('--network_lat', type=float, default=0.5e-6, help=argparse.SUPPRESS)
_pm_group.add_argument('--network_alg', type=str, default="vdg", help=argparse.SUPPRESS)
