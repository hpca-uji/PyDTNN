#!/usr/bin/env python

"""
PyDTNN print in convdirect format script
"""

#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

# from __future__ import print_function

from pydtnn.model import Model
from pydtnn.parser import parser

# Parse options
params = parser.parse_args()
# Create model
model = Model(**vars(params))
# Call print_in_convdirect_format
# print(f"#Model: {model.model_name}")
model.print_in_convdirect_format()

# Examples
# --------

# Print the layers of ResNet50v15 for ImageNet in convdirect input format:
# pydtnn/scripts/print_in_convdirect_format.py --model=resnet50v15_imagenet

# Print the memory required for the Im2Row transformation of each layer with the default batch size and float32
# pydtnn/scripts/print_in_convdirect_format.py --model=resnet50v15_imagenet \
#   | awk '!/#/ {print $1 " "  $5 * $6 * $7 * $8 * $9 * $10 * 4 / 1024 / 1024}' | sort -k 2  -g

# Print the memory required for the Im2Row transformation of each layer with batch size of 1 and float32
# pydtnn/scripts/print_in_convdirect_format.py --model=resnet50v15_imagenet \
#   | awk '!/#/ {print $1 " "  $6 * $7 * $8 * $9 * $10 * 4 / 1024 / 1024}' | sort -k 2  -g

