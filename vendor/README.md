# External dependencies

```sh
# Configuration
SRC="$PWD/vendor"
PREFIX="$PWD/vendor/build"
NPROC=$(nproc)
```

## BLIS
Source: <https://github.com/flame/blis>

Dependencies: `make gcc`

```sh
# Configuration
BLIS_SRC="$SRC/blis"
BLIS_PREFIX="$PREFIX/blis"

# Source
# git clone https://github.com/flame/blis.git "$BLIS_SRC"
git submodule update --init "$BLIS_SRC"
cd "$BLIS_SRC"
git checkout 0.7.0

# Compile
./configure --prefix="$BLIS_PREFIX" --enable-cblas auto
make -j "$NPROC"

# Install
mkdir -p "$BLIS_PREFIX"
make install
export LD_LIBRARY_PATH="$BLIS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## TVM
Source: <https://github.com/apache/tvm>

Dependencies: `python3 cmake gcc llvm-dev` and virutal Python environment

```sh
# Configuration
TVM_SRC="$SRC/tvm"
TVM_PREFIX="$PREFIX/tvm"

# Source
# git clone --recursive https://github.com/apache/tvm.git "$TVM_SRC"
git submodule update --init --recursive "$TVM_SRC"
cd "$TVM_SRC"
git checkout 43e9c275b6e85d7631e54c8468b49b4706cd674a

# Compile
mkdir -p ./build
cd ./build
cp ../cmake/config.cmake .
cmake -D CMAKE_INSTALL_PREFIX="$TVM_PREFIX" ..
cmake --build . --parallel "$NPROC"

# Install
mkdir -p "$TVM_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$TVM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd ..
pip install numpy psutil
pip install ./3rdparty/tvm-ffi
pip install ./python
```

## convGemm
Source: <https://github.com/hpca-uji/convGemm>

Dependencies: `cmake gcc` and `blis`

```sh
# Configuration
BLIS_PREFIX="$PREFIX/blis"
GEMM_SRC="$SRC/convGemm"
GEMM_PREFIX="$PREFIX/convGemm"

# Source
# git clone https://github.com/hpca-uji/convGemm.git "$GEMM_SRC"
git submodule update --init "$GEMM_SRC"
cd "$GEMM_SRC"
git checkout 1ebea3c77cd961cb207f5964025733913765b0e6

# Compile
cd ./build
cmake -D CMAKE_PREFIX_PATH="$BLIS_PREFIX" -D CMAKE_INSTALL_PREFIX="$GEMM_PREFIX" ..
cmake --build . --parallel "$NPROC"

# Install
mkdir -p "$GEMM_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$GEMM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## convWinograd
Source: <https://github.com/hpca-uji/convWinograd>

Dependencies: `cmake gcc` and `blis`

```sh
# Configuration
BLIS_PREFIX="$PREFIX/blis"
WINOGRAD_SRC="$SRC/convWinograd"
WINOGRAD_PREFIX="$PREFIX/convWinograd"

# Source
# git clone https://github.com/hpca-uji/convWinograd.git "$WINOGRAD_SRC"
git submodule update --init "$WINOGRAD_SRC"
cd "$WINOGRAD_SRC"
git checkout fc2d5af8d0ee551e508b97082ee7aab3bbff0244

# Compile
cd ./build
cmake -D BLA_VENDOR=FLAME -D CMAKE_PREFIX_PATH="$BLIS_PREFIX" -D CMAKE_INSTALL_PREFIX="$WINOGRAD_PREFIX" ..
cmake --build . --parallel "$NPROC"

# Install
mkdir -p "$WINOGRAD_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$WINOGRAD_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## convDirect
Source: <https://github.com/hpca-uji/convDirect>

Dependencies: `cmake gcc` and `blis tvm convGemm`

```sh
# Configuration
BLIS_PREFIX="$PREFIX/blis"
TVM_PREFIX="$PREFIX/tvm"
DIRECT_SRC="$SRC/convDirect"
DIRECT_PREFIX="$PREFIX/convDirect"

# Source
# git clone --recursive https://github.com/hpca-uji/convDirect.git "$DIRECT_SRC"
git submodule update --init --recursive "$DIRECT_SRC"
cd "$DIRECT_SRC"
git checkout 25937a6b3e06cf06089e7403798547c31528cba3

# Compile
cd ./build
cmake -D CMAKE_PREFIX_PATH="$BLIS_PREFIX;$TVM_PREFIX" -D CMAKE_INSTALL_PREFIX="$DIRECT_PREFIX" ..
cmake --build . --parallel "$NPROC"

# Install
mkdir -p "$DIRECT_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$DIRECT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## OpenFHE
Source: <https://github.com/openfheorg/openfhe-development>

Dependencies: `cmake gcc`

```sh
# Configuration
OFHE_PREFIX="$PREFIX/openfhe"
OFHE_SRC="$SRC/openfhe"

# Source
# git clone https://github.com/openfheorg/openfhe-development.git "$OFHE_SRC"
git submodule update --init "$OFHE_SRC"
cd "$OFHE_SRC"
git checkout v1.4.2

# Compile
mkdir -p ./build
cd ./build
cmake -D CMAKE_INSTALL_PREFIX="$OFHE_PREFIX" ..
cmake --build . --parallel "$NPROC"

# Install
mkdir -p "$OFHE_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$OFHE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## OpenFHE-Python
Source: <https://github.com/openfheorg/openfhe-python>

Dependencies: `python3 cmake gcc` and virutal Python environment

```sh
# Configuration
OFHE_PREFIX="$PREFIX/openfhe"
OFHEPY_SRC="$SRC/openfhe-python"

# Source
# git clone https://github.com/openfheorg/openfhe-python.git "$OFHEPY_SRC"
git submodule update --init "$OFHEPY_SRC"
cd "$OFHEPY_SRC"
git checkout v1.4.2.0
pip install pybind11[global]

# Compile
mkdir -p ./build
cd ./build
cmake -D CMAKE_PREFIX_PATH="$OFHE_PREFIX" -D CMAKE_INSTALL_PREFIX="$OFHEPY_SRC/openfhe" ..
cmake --build . --parallel "$NPROC"
cat <<EOF > "$OFHEPY_SRC/pyproject.toml"
[project]
name = "openfhe"
version = "1.4.2"
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[tool.setuptools]
packages.find.include = ["openfhe", "openfhe.*"]
package-data.openfhe = ["*"]
EOF
cmake --install .

# Install
pip install "$OFHEPY_SRC"
```

## uArchFHE
Source: <https://github.com/darwinquezada/he_hpc>

Dependencies: `python3 libgmp-dev libntl-dev libbz2-dev` and virutal Python environment

```sh
# Configuration
UAFHE_PREFIX="$PREFIX/uarchfhe"
UAFHE_SRC="$SRC/uarchfhe"

# Source
# git clone https://github.com/darwinquezada/he_hpc.git "$UAFHE_SRC"
git submodule update --init "$UAFHE_SRC"
cd "$UAFHE_SRC"
git checkout 7970d0dfad5b74939da492cf61c5d9c4a9753c19

# Install
pip install "$UAFHE_SRC/crates/fhe_py_binding"
```

## PolyHE
Source: <https://github.com/hpca-uji/polyhe>

Dependencies: `python3` and virutal Python environment

```sh
# Configuration
POLYHE_SRC="$SRC/polyhe"

# Source
# git clone https://github.com/hpca-uji/polyhe.git "$POLYHE_SRC"
git submodule update --init "$POLYHE_SRC"
cd "$POLYHE_SRC"
git checkout 05f2a4c168f370d2ea5c1781bed0758f63c26687

# Install
pip install "$POLYHE_SRC"
```

## pympi
Source: <https://github.com/hpca-uji/pympi>

Dependencies: `python3` and virutal Python environment

```sh
# Configuration
PYMPI_SRC="$SRC/pympi"

# Source
# git clone https://github.com/hpca-uji/pympi.git "$PYMPI_SRC"
git submodule update --init "$PYMPI_SRC"
cd "$PYMPI_SRC"
git checkout a1bddf15ce0675a7748ec4ebf9fa2c779e0c9285

# Install
pip install "$PYMPI_SRC"
```