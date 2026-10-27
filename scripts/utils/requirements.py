#!/usr/bin/env python3
"""Generate a requirements file with all extras"""

from importlib.metadata import metadata
from pydtnn import package_name

for requirement in metadata(package_name).get_all("Requires-Dist", []):
    requirement = requirement.split(";", 1)[0]
    parts = requirement.split("[", 1)

    if parts[0] == package_name:
        continue

    print(requirement)
