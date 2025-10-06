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
```

## Local
```sh
git clone https://github.com/flame/blis.git
cd ./blis
git checkout 0.7.0
mkdir -p ~/opt/blis
./configure --prefix=~/opt/blis auto
make -j
make install
```

---

# convGemm

Source: <https://github.com/hpca-uji/convGemm>

## Global
```sh
git clone https://github.com/hpca-uji/convGemm.git
cd ./convGemm
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
cd ./build
cmake ..
make -j
sudo make install
export LD_LIBRARY_PATH=/usr/local/lib
```

## Local
```sh
mkidr ~/opt/convGemm
git clone https://github.com/hpca-uji/convGemm.git
cd ./convGemm
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
cd ./build
cmake -D CMAKE_PREFIX_PATH=~/opt/blis -D CMAKE_INSTALL_PREFIX=~/opt/convGemm ..
make -j
make install
export LD_LIBRARY_PATH=~/opt/convGemm/lib:~/opt/blis/lib
```

---

# Execute

## All
```sh
python -m unittest -v pydtnn.tests
```

## Specific
```sh
python -m unittest -v pydtnn.tests.${name_test}
```
