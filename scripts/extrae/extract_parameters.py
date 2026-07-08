#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extracts parameters from PyDTNN run_model_dataset.sh files."""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2022 Universitat Jaume I
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

import getopt
import os
import pathlib
import re
import sys
from collections import defaultdict

from rich import box
from rich.console import Console
from rich.table import Table


def my_help() -> None:
    """Print the command line usage help."""
    print("""Usage: extract_parameters.py [OPTION]...

Extracts parameters from run_model_dataset.sh files

Options:
    -v, --VERBOSE      increment the output verbosity.
    -h, --help         display this help and exit.

Please, report bugs to <barrachi@uji.es>.
""")


def log(text: str) -> None:
    """Log a message to stderr."""
    sys.stderr.write(">>> %s\n" % text)


def error(text: str) -> None:
    """Report an error message and exit."""
    sys.stderr.write("ERROR: %s\n" % text)
    sys.exit(-1)


# Global command line parameters
VERBOSE = 0
SCRIPT_PATH = pathlib.Path(__file__).parent.absolute()


def get_opts() -> None:
    """Read command line options."""
    global VERBOSE
    optlist, args = getopt.getopt(sys.argv[1:], "hv", ["VERBOSE", "help"])
    for opt, arg in optlist:
        if opt in ("-h", "--help"):
            my_help()
            sys.exit()
        elif opt in ("-v", "--verbosity"):
            VERBOSE = 1


def extract_parameters() -> None:  # noqa: C901
    """Parse shell scripts to extract and display model configuration parameters."""
    models_files = {
        "alexnet_cifar10": "run_alexnet_cifar10.sh",
        "alexnet_imagenet": "run_alexnet_imagenet.sh",
        "vgg16_cifar10": "run_vgg_cifar10.sh",
        "vgg16_imagenet": "run_vgg_imagenet.sh",
        "resnet34_cifar10": "run_resnet_cifar10.sh",
        "resnet34_imagenet": "run_resnet_imagenet.sh",
        "resnet50_cifar10": "run_resnet_cifar10.sh",
        "resnet50_imagenet": "run_resnet_imagenet.sh",
        "resnet50v15_cifar10": "run_resnet_cifar10.sh",
        "resnet50v15_imagenet": "run_resnet_imagenet.sh",
        "densenet121_cifar10": "run_densenet_cifar10.sh",
        "densenet121_imagenet": "run_densenet_imagenet.sh",
        "googlenet_cifar10": "run_inception_cifar10.sh",
        "googlenet_imagenet": "run_inception_imagenet.sh",
    }
    parameters_with_short_values = defaultdict(lambda: defaultdict(lambda: ""))
    parameters_with_long_values = defaultdict(lambda: defaultdict(lambda: ""))
    param_keys = list(parameters_with_short_values.keys())
    param_keys.sort()

    # Manually overwritten parameters
    parameters_overwritten = defaultdict(lambda: defaultdict(lambda: "-"))
    for model in models_files.keys():
        parameters_overwritten["model"][model] = model
        parameters_overwritten["evaluate"][model] = "True"
        parameters_overwritten["augment_flip"][model] = "False"
        parameters_overwritten["augment_mask"][model] = "False"
        parameters_overwritten["enable_cudnn"][model] = "False"
    parameters_overwritten["optimizer"]["vgg16_imagenet"] = "sgd"
    parameters_overwritten["learning_rate"]["vgg16_imagenet"] = "0.01"
    parameters_overwritten["test_as_validation"]["vgg16_imagenet"] = "False"
    parameters_overwritten["test_as_validation"]["resnet34_imagenet"] = "False"

    # Removing schedulers and associated parameters
    for model in models_files.keys():
        parameters_overwritten["schedulers"][model] = ""
        parameters_overwritten["early_stopping_metric"][model] = ""
        parameters_overwritten["reduce_lr_on_plateau_metric"][model] = ""
        parameters_overwritten["stop_at_loss_metric"][model] = ""

    # Patterns
    command_pattern = re.compile(r"[^#]*.*pydtnn-benchmark")
    parameter_pattern = re.compile(r"--([^=]+)=([^ ]+)")
    ignore_parameters = ("dataset_path", "parallel_data", "parallel_pipeline", "history_file")

    # Extract the parameters
    for model, file in models_files.items():
        with open(os.path.join(SCRIPT_PATH, file), "r") as f:
            wait_for_it = True
            for line in f.readlines():
                if wait_for_it:
                    if command_pattern.search(line):
                        wait_for_it = False
                    continue
                line = line.strip()
                if parameter_pattern.search(line):
                    match = parameter_pattern.search(line)
                    assert match, "Line format wrong!"
                    param, value = match.groups()
                    if param in ignore_parameters:
                        continue
                    if len(value) > 500:  # 9:
                        if param in parameters_with_short_values.keys():
                            parameters_with_long_values[param] = parameters_with_short_values[param]
                            parameters_with_short_values.pop(param)
                        parameters_with_long_values[param][model] = value
                    else:
                        if param in parameters_with_long_values.keys():
                            parameters_with_long_values[param][model] = value
                        else:
                            parameters_with_short_values[param][model] = value
                else:
                    break

    c = Console()
    c.print("#")
    c.print("# DO NOT EDIT THIS FILE!")
    c.print("#")
    c.print(
        "# It has been automatically generated with 'extract_parameters.py' from the next files:"
    )
    c.print("#")
    for model, file in models_files.items():
        c.print(f"#   {model} <- {file}")
    c.print("#")
    c.print(
        "# Those parameters that have been edited on the script have been prepended with an '*'."
    )
    c.print("#")

    t = Table(box=box.ASCII)
    t.add_column("parameter")
    for key in models_files.keys():
        word1, word2 = key.split("_")
        t.add_column(f"{word1[0].upper()}{word2[0].upper()}")

    for param in param_keys:
        value_for = parameters_with_short_values[param]
        row = [
            param,
        ]
        for model in models_files.keys():
            if (
                parameters_overwritten[param][model] != "-"
                and value_for[model] != parameters_overwritten[param][model]
            ):
                row.append(f"[bold]*{parameters_overwritten[param][model]}")
            else:
                row.append(value_for[model])
        t.add_row(*row)
    c.print(t)

    for param, value_for in parameters_with_long_values.items():
        t = Table(box=box.ASCII)
        t.add_column("model_dataset")
        t.add_column(param)
        for model in models_files:
            t.add_row(model, value_for[model])
        c.print(t)

    for param, value_for in parameters_with_short_values.items():
        row = [
            param,
        ]
        for model in models_files.keys():
            if (
                parameters_overwritten[param][model] != "-"
                and value_for[model] != parameters_overwritten[param][model]
            ):
                row.append(f"{parameters_overwritten[param][model]}")
            else:
                row.append(value_for[model])
        print(";".join(row))


def main() -> None:
    """Do the work (main function, called when not imported)."""
    get_opts()
    # Main part of the application
    extract_parameters()


if __name__ == "__main__":
    main()
