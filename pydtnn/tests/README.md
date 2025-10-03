BLIS
https://github.com/flame/blis?tab=readme-ov-file#getting-started

git clone https://github.com/flame/blis.git
cd ./blis
git checkout 0.7.0
./configure auto
make [-j]
make install

convGemm
https://github.com/hpca-uji/convGemm.git

git clone https://github.com/hpca-uji/convGemm.git
cd ./convGemm
git checkout cd1f2e8d7e5079aa23f6482b115377d40fe6b7bc
cd ./build
cmake ..
make -j
make install

export LD_LIBRARY_PATH=/usr/local/lib


---
python -m unittest -v pydtnn.tests
