# ============================================================================
# Configuration
# ============================================================================

SRC            := $(CURDIR)
PREFIX         := $(CURDIR)/build
NPROC          := $(shell nproc)

BLIS_SRC       := $(SRC)/blis
BLIS_PREFIX    := $(PREFIX)/blis

TVM_SRC        := $(SRC)/tvm
TVM_PREFIX     := $(PREFIX)/tvm

GEMM_SRC       := $(SRC)/convGemm
GEMM_PREFIX    := $(PREFIX)/convGemm

WINOGRAD_SRC   := $(SRC)/convWinograd
WINOGRAD_PREFIX:= $(PREFIX)/convWinograd

DIRECT_SRC     := $(SRC)/convDirect
DIRECT_PREFIX  := $(PREFIX)/convDirect

OFHE_SRC       := $(SRC)/openfhe
OFHE_PREFIX    := $(PREFIX)/openfhe

OFHEPY_SRC     := $(SRC)/openfhe-python

UAFHE_SRC      := $(SRC)/uarchfhe
UAFHE_PREFIX   := $(PREFIX)/uarchfhe

POLYHE_SRC     := $(SRC)/polyhe

PYMPI_SRC      := $(SRC)/pympi

export LD_LIBRARY_PATH := \
	$(BLIS_PREFIX)/lib:\
	$(TVM_PREFIX)/lib:\
	$(GEMM_PREFIX)/lib:\
	$(WINOGRAD_PREFIX)/lib:\
	$(DIRECT_PREFIX)/lib:\
	$(OFHE_PREFIX)/lib:\
	$(LD_LIBRARY_PATH)

# ============================================================================
# Targets
# ============================================================================

.PHONY: all \
	blis tvm convgemm convwinograd convdirect \
	openfhe openfhe-python uarchfhe polyhe pympi \
	config clean

all: \
	blis \
	tvm \
	convgemm \
	convwinograd \
	convdirect \
	openfhe \
	openfhe-python \
	uarchfhe \
	polyhe \
	pympi

# ============================================================================
# BLIS
# ============================================================================

blis:
	git submodule update --init "$(BLIS_SRC)"
	cd "$(BLIS_SRC)" && \
		git checkout 0.7.0 && \
		./configure --prefix="$(BLIS_PREFIX)" --enable-cblas auto && \
		$(MAKE) -j "$(NPROC)" && \
		mkdir -p "$(BLIS_PREFIX)" && \
		$(MAKE) install

# ============================================================================
# TVM
# ============================================================================

tvm:
	git submodule update --init --recursive "$(TVM_SRC)"
	cd "$(TVM_SRC)" && \
		git checkout 43e9c275b6e85d7631e54c8468b49b4706cd674a && \
		mkdir -p build && \
		cd build && \
		cp ../cmake/config.cmake . && \
		cmake -D CMAKE_INSTALL_PREFIX="$(TVM_PREFIX)" .. && \
		cmake --build . --parallel "$(NPROC)" && \
		mkdir -p "$(TVM_PREFIX)" && \
		cmake --install .
	cd "$(TVM_SRC)" && \
		pip install numpy psutil && \
		pip install ./3rdparty/tvm-ffi && \
		pip install ./python

# ============================================================================
# convGemm
# ============================================================================

convgemm: blis
	git submodule update --init "$(GEMM_SRC)"
	cd "$(GEMM_SRC)" && \
		git checkout 1ebea3c77cd961cb207f5964025733913765b0e6 && \
		cd build && \
		cmake \
			-D CMAKE_PREFIX_PATH="$(BLIS_PREFIX)" \
			-D CMAKE_INSTALL_PREFIX="$(GEMM_PREFIX)" \
			.. && \
		cmake --build . --parallel "$(NPROC)" && \
		mkdir -p "$(GEMM_PREFIX)" && \
		cmake --install .

# ============================================================================
# convWinograd
# ============================================================================

convwinograd: blis
	git submodule update --init "$(WINOGRAD_SRC)"
	cd "$(WINOGRAD_SRC)" && \
		git checkout fc2d5af8d0ee551e508b97082ee7aab3bbff0244 && \
		cd build && \
		cmake \
			-D BLA_VENDOR=FLAME \
			-D CMAKE_PREFIX_PATH="$(BLIS_PREFIX)" \
			-D CMAKE_INSTALL_PREFIX="$(WINOGRAD_PREFIX)" \
			.. && \
		cmake --build . --parallel "$(NPROC)" && \
		mkdir -p "$(WINOGRAD_PREFIX)" && \
		cmake --install .

# ============================================================================
# convDirect
# ============================================================================

convdirect: blis tvm convgemm
	git submodule update --init --recursive "$(DIRECT_SRC)"
	cd "$(DIRECT_SRC)" && \
		git checkout 25937a6b3e06cf06089e7403798547c31528cba3 && \
		cd build && \
		cmake \
			-D CMAKE_PREFIX_PATH="$(BLIS_PREFIX);$(TVM_PREFIX)" \
			-D CMAKE_INSTALL_PREFIX="$(DIRECT_PREFIX)" \
			.. && \
		cmake --build . --parallel "$(NPROC)" && \
		mkdir -p "$(DIRECT_PREFIX)" && \
		cmake --install .

# ============================================================================
# OpenFHE
# ============================================================================

openfhe:
	git submodule update --init "$(OFHE_SRC)"
	cd "$(OFHE_SRC)" && \
		git checkout v1.4.2 && \
		mkdir -p build && \
		cd build && \
		cmake -D CMAKE_INSTALL_PREFIX="$(OFHE_PREFIX)" .. && \
		cmake --build . --parallel "$(NPROC)" && \
		mkdir -p "$(OFHE_PREFIX)" && \
		cmake --install .

# ============================================================================
# OpenFHE Python
# ============================================================================

openfhe-python: openfhe
	git submodule update --init "$(OFHEPY_SRC)"
	cd "$(OFHEPY_SRC)" && \
		git checkout v1.4.2.0

	pip install pybind11[global]

	cd "$(OFHEPY_SRC)" && \
		mkdir -p build && \
		cd build && \
		cmake \
			-D CMAKE_PREFIX_PATH="$(OFHE_PREFIX)" \
			-D CMAKE_INSTALL_PREFIX="$(OFHEPY_SRC)/openfhe" \
			.. && \
		cmake --build . --parallel "$(NPROC)"

	cat <<EOF > "$(OFHEPY_SRC)/pyproject.toml"
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

	cd "$(OFHEPY_SRC)/build" && \
		cmake --install .

	pip install "$(OFHEPY_SRC)"

# ============================================================================
# uArchFHE
# ============================================================================

uarchfhe:
	git submodule update --init "$(UAFHE_SRC)"
	cd "$(UAFHE_SRC)" && \
		git checkout 7970d0dfad5b74939da492cf61c5d9c4a9753c19

	pip install "$(UAFHE_SRC)/crates/fhe_py_binding"

# ============================================================================
# PolyHE
# ============================================================================

polyhe:
	git submodule update --init "$(POLYHE_SRC)"
	cd "$(POLYHE_SRC)" && \
		git checkout 804e4d48fbf792cfe61c1a8f283ed6ef1e907e7f

	pip install "$(POLYHE_SRC)"

# ============================================================================
# PyMPI
# ============================================================================

pympi:
	git submodule update --init "$(PYMPI_SRC)"
	cd "$(PYMPI_SRC)" && \
		git checkout a1bddf15ce0675a7748ec4ebf9fa2c779e0c9285

	pip install "$(PYMPI_SRC)"

# ============================================================================
# Configuration
# ============================================================================

config:
	echo LD_LIBRARY_PATH="$(LD_LIBRARY_PATH)""


# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf "$(PREFIX)"
