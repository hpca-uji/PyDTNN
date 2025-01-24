#!/usr/bin/env python3
# setup venv and pip remotely

import sys
sys.path.pop(0)

import venv
import shlex
import argparse
import subprocess
from pathlib import Path
from urllib.request import urlopen


def main():
    # meta
    name = Path(__file__).name
    desc = Path(__file__).read_text().splitlines()[1]
    pip_url = "https://bootstrap.pypa.io/get-pip.py"

    # parser
    parser = argparse.ArgumentParser(prog=name, description=desc)
    parser.add_argument("env_dir", type=Path, help="conventionally .venv")
    parser.print_usage = parser.print_help
    config, args = parser.parse_known_args()

    # venv
    venv.create(config.env_dir, symlinks=True, with_pip=False)
    activate = config.env_dir / "bin/activate"
    source = shlex.join([".", str(activate)])

    # pip
    pip = config.env_dir / "pip.py"
    with urlopen(pip_url) as response:
        pip.write_bytes(response.read())
    pip = shlex.join(["python", str(pip), *args])
    subprocess.run(f"{source}; {pip}", shell=True, check=True, stdout=subprocess.DEVNULL)

    # activate
    print("Activate with:", file=sys.stderr)
    print(source, file=sys.stderr)


if __name__ == "__main__":
    main()
