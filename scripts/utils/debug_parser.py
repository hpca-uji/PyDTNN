#!/usr/bin/env python3
# Print all parser options

from pydtnn.utils.parser import ArgumentParser

for k, v in vars(ArgumentParser().parse_args()).items():
    print(f"{k}: {v.__class__.__name__} = {v!r}")
