#!/usr/bin/env python3
"""Get all package extras"""

from importlib.metadata import metadata
from pydtnn import package_name

for extra in metadata(package_name).get_all("Provides-Extra", []):
    print(extra)
