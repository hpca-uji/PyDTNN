#!/usr/bin/env python3

"""
Exports a PyDTNN dataset.
"""

from pydtnn.model import Model
from pydtnn.parser import PydtnnArgumentParser

parser = PydtnnArgumentParser()
parser.add_argument("--export_weights", type=str, default="1")
model = Model(**parser.to_dict())
weights = list(map(float, model.export_weights.split(",")))
model.dataset.export(weights)
