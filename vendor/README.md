# External dependencies
**Execute each from project root**, modify paths as desired.

```sh
# Configuration
SRC="$PWD/vendor"
PREFIX="$PWD/vendor/build"
NPROC=$(nproc)
```

## BLIS
Source: <https://github.com/flame/blis>

Dependencies: `gcc make`

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

Dependencies: `gcc cmake llvm-dev python3` and virutal Python environment

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

Dependencies: `gcc cmake` and `blis`

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

Dependencies: `gcc cmake` and `blis`

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

Dependencies: `gcc cmake` and `blis tvm convGemm`

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
git checkout 888402fc45df89f8a055dc4575ef53a6b35ea502

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

Dependencies: `gcc cmake`

```sh
# Configuration
FHE_PREFIX="$PREFIX/openfhe"
FHE_SRC="$SRC/openfhe"

# Source
# git clone https://github.com/openfheorg/openfhe-development.git "$FHE_SRC"
git submodule update --init "$FHE_SRC"
cd "$FHE_SRC"
git checkout v1.4.2

# Compile
mkdir -p ./build
cd ./build
cmake -D CMAKE_INSTALL_PREFIX="$FHE_PREFIX" ..
cmake --build . --parallel "$NPROC"

# Install
mkdir -p "$FHE_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$FHE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## OpenFHE-Python
Source: <https://github.com/openfheorg/openfhe-python>

Dependencies: `gcc cmake python3` and virutal Python environment

```sh
# Configuration
FHE_PREFIX="$PREFIX/openfhe"
PYFHE_SRC="$SRC/openfhe-python"
PYFHE_PREFIX="$PREFIX/openfhe-python"

# Source
# git clone https://github.com/openfheorg/openfhe-python.git "$PYFHE_SRC"
git submodule update --init "$PYFHE_SRC"
cd "$PYFHE_SRC"
git checkout v1.4.2.0
pip install pybind11[global]

# Compile
mkdir -p ./build
cd ./build
cmake -D CMAKE_PREFIX_PATH="$FHE_PREFIX" -D CMAKE_INSTALL_PREFIX=openfhe ..
cmake --build . --parallel "$NPROC"
cat <<EOF > pyproject.toml
[project]
name = "openfhe"
version = "1.4.2"
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[tool.setuptools.package-data]
openfhe = ["*"]
EOF

# Install
mkdir -p "$PYFHE_PREFIX"
cmake --install .
cp -at "$PYFHE_PREFIX" pyproject.toml openfhe
pip install "$PYFHE_PREFIX"
```

## net-queue
Source: <https://github.com/hpca-uji/net-queue>

Dependencies: `python3` and virutal Python environment

```sh
# Configuration
NQ_SRC="$SRC/net-queue"

# Source
# git clone https://github.com/hpca-uji/net-queue.git "$NQ_SRC"
git submodule update --init "$NQ_SRC"
cd "$NQ_SRC"
git checkout 283540374e5b0cff7758b7549dd0a67eee2d590b

# Install
pip install .
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
git checkout f8da55d7d79e0e048ae2fcb8008b9e4e9ad6dc38

# Install
pip install .
```