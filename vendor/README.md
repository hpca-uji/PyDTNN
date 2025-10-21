# External dependencies
Execute each from project root, modify paths as desired.

```sh
# Configuration
SRC="$PWD/vendor"
PREFIX="$HOME/opt"
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
git submodule update --init vendor/blis
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
git submodule update --init --recursive vendor/tvm
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
GEMM_PATCH="$SRC/convGemm.patch"
GEMM_PREFIX="$PREFIX/convGemm"

# Source
# git clone https://github.com/hpca-uji/convGemm.git "$GEMM_SRC"
git submodule update --init vendor/convGemm
cd "$GEMM_SRC"
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
git apply "$GEMM_PATCH"

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
WINOGRAD_PATCH="$SRC/convWinograd.patch"
WINOGRAD_PREFIX="$PREFIX/convWinograd"

# Source
# git clone https://github.com/hpca-uji/convWinograd.git "$WINOGRAD_SRC"
git submodule update --init vendor/convWinograd
cd "$WINOGRAD_SRC"
git checkout 0a1ca8b22f9ee12d4006f28c16c0e6f6e88ad939
git apply "$WINOGRAD_PATCH"

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
GEMM_SRC="$SRC/convGemm"
DIRECT_SRC="$SRC/convDirect"
DIRECT_PATCH="$SRC/convDirect.patch"
DIRECT_PREFIX="$PREFIX/convDirect"

# Source
# git clone --recursive https://github.com/hpca-uji/convDirect.git "$DIRECT_SRC"
git submodule update --init --recursive vendor/convDirect
cd "$DIRECT_SRC"
git checkout 352dadb1990fd882b16f10b22fcb842d3856be57
git apply "$DIRECT_PATCH"
rm -r ./src/convGemm
ln -s "$GEMM_SRC" ./src/convGemm

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
git submodule update --init vendor/openfhe
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
git submodule update --init vendor/openfhe-python
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
git submodule update --init vendor/net-queue
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
git submodule update --init vendor/pympi
cd "$PYMPI_SRC"
git checkout f8da55d7d79e0e048ae2fcb8008b9e4e9ad6dc38

# Install
pip install .
```