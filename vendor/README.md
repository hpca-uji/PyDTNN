# External dependencies

## BLIS
Source: <https://github.com/flame/blis>
Dependencies: `gcc make`

```sh
BLIS_PREFIX=~/opt/blis
git clone https://github.com/flame/blis.git
cd ./blis
git checkout 0.7.0
./configure --prefix="$BLIS_PREFIX" --enable-cblas auto
make -j
mkdir -p "$BLIS_PREFIX"
make install
export LD_LIBRARY_PATH="$BLIS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## TVM
Source: <https://github.com/apache/tvm>
Dependencies: `gcc cmake llvm-dev python3`

```sh
TVM_PREFIX=~/opt/tvm
git clone --recursive https://github.com/apache/tvm.git
cd ./tvm
git checkout 43e9c275b6e85d7631e54c8468b49b4706cd674a
mkdir ./build
cd ./build
cp ../cmake/config.cmake .
cmake -D CMAKE_INSTALL_PREFIX="$TVM_PREFIX" ..
cmake --build . --parallel
mkdir -p "$TVM_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$TVM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TVM_LIBRARY_PATH="$TVM_PREFIX"
cd ..
pip install numpy psutil
pip install ./3rdparty/tvm-ffi
pip install ./python
```

## convGemm
Source: <https://github.com/hpca-uji/convGemm>
Dependencies: `gcc cmake`

```sh
PATCH_SRC=~/src/patch
BLIS_PREFIX=~/opt/blis
GEMM_PREFIX=~/opt/convGemm
git clone https://github.com/hpca-uji/convGemm.git
cd ./convGemm
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
git apply "$PATCH_SRC/convGemm.patch"
cd ./build
cmake -D CMAKE_PREFIX_PATH="$BLIS_PREFIX" -D CMAKE_INSTALL_PREFIX="$GEMM_PREFIX" ..
cmake --build . --parallel
mkdir -p "$GEMM_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$GEMM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## convWinograd
Source: <https://github.com/hpca-uji/convWinograd>
Dependencies: `gcc cmake`

```sh
PATCH_SRC=~/src/patch
BLIS_PREFIX=~/opt/blis
WINOGRAD_PREFIX=~/opt/convWinograd
git clone https://github.com/hpca-uji/convWinograd.git
cd ./convWinograd
git checkout 0a1ca8b22f9ee12d4006f28c16c0e6f6e88ad939
git apply "$PATCH_SRC/convWinograd.patch"
cd ./build
cmake -D BLA_VENDOR=FLAME -D CMAKE_PREFIX_PATH="$BLIS_PREFIX" -D CMAKE_INSTALL_PREFIX="$WINOGRAD_PREFIX" ..
cmake --build . --parallel
mkdir -p "$WINOGRAD_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$WINOGRAD_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## convDirect
Source: <https://github.com/hpca-uji/convDirect>
Dependencies: `gcc cmake`

```sh
PATCH_SRC=~/src/patch
BLIS_PREFIX=~/opt/blis
GEMM_SRC=~/src/convGemm
TVM_PREFIX=~/opt/tvm
DIRECT_PREFIX=~/opt/convDirect
git clone --recursive https://github.com/hpca-uji/convDirect.git
cd ./convDirect
git checkout 352dadb1990fd882b16f10b22fcb842d3856be57
git apply "$PATCH_SRC/convDirect.patch"
rm -r ./src/convGemm
git submodule set-url src/convGemm "$GEMM_SRC"
git submodule update --init src/convGemm
cd ./build
cmake -D CMAKE_PREFIX_PATH="$BLIS_PREFIX;$TVM_PREFIX" -D CMAKE_INSTALL_PREFIX="$DIRECT_PREFIX" ..
cmake --build . --parallel
mkdir -p "$DIRECT_PREFIX"
cmake --install .
export LD_LIBRARY_PATH="$DIRECT_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```
