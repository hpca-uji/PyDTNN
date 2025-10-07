# Tests

In order to run all the test, it is necessary to install both *Bliss* and *convGemm* libraries. In the following section the installation steps of both libraries is detailed.

---

# BLIS
Source: <https://github.com/flame/blis>

## Global
```sh
git clone https://github.com/flame/blis.git
cd ./blis
git checkout 0.7.0
./configure auto
make -j
sudo make install
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Local
```sh
BLIS_PREFIX=~/opt/blis
git clone https://github.com/flame/blis.git
cd ./blis
git checkout 0.7.0
mkdir -p "$BLIS_PREFIX"
./configure --prefix="$BLIS_PREFIX" auto
make -j
make install
export LD_LIBRARY_PATH="$BLIS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

# convGemm

Source: <https://github.com/hpca-uji/convGemm>

## Patches
```c
// src/gemm_blis.h
#include <omp.h>
```

```c
// tests/test_base.h
#include <stdbool.h>
```

## Global
```sh
git clone https://github.com/hpca-uji/convGemm.git
cd ./convGemm
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
cd ./build
cmake ..
make -j
sudo make install
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Local
```sh
BLIS_PREFIX=~/opt/blis
GEMM_PREFIX=~/opt/convGemm
git clone https://github.com/hpca-uji/convGemm.git
cd ./convGemm
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
cd ./build
cmake -D CMAKE_PREFIX_PATH="$BLIS_PREFIX" -D CMAKE_INSTALL_PREFIX="$GEMM_PREFIX" ..
make -j
make install
export LD_LIBRARY_PATH="$GEMM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

# Execute

## All
```sh
python -m unittest -v pydtnn.tests
```
This command will execute all tests except the `conv2d_conv_gemm_slow` ones (it is very slow). To execute this test use the following command:

```sh
python -m unittest -v pydtnn.tests.conv2d_conv_gemm_slow.Conv2DConvGemmSlowTestCase.test_forward_backward_multiple_params
```

## Specific
```sh
python -m unittest -v pydtnn.tests.${name_test}
```
