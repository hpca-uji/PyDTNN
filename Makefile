# ============================================================================
# Configuration
# ============================================================================

.DEFAULT_GOAL := pydtnn-develop
SHELL := $(shell which bash)
PROCS := $(shell nproc)

SRC := $(CURDIR)/vendor
DST := $(CURDIR)/build

BLIS_VER := 68b88aca6692c75a9f686187e6c4a4e196ae60a9
BLIS_SRC := $(SRC)/blis
BLIS_DST := $(DST)/blis

TVM_VER := 43e9c275b6e85d7631e54c8468b49b4706cd674a
TVM_SRC := $(SRC)/tvm
TVM_DST := $(DST)/tvm

CONVGEMM_VER := 1ebea3c77cd961cb207f5964025733913765b0e6
CONVGEMM_SRC := $(SRC)/convGemm
CONVGEMM_DST := $(DST)/convGemm

CONVWINOGRAD_VER := fc2d5af8d0ee551e508b97082ee7aab3bbff0244
CONVWINOGRAD_SRC := $(SRC)/convWinograd
CONVWINOGRAD_DST := $(DST)/convWinograd

CONVDIRECT_VER := 25937a6b3e06cf06089e7403798547c31528cba3
CONVDIRECT_SRC := $(SRC)/convDirect
CONVDIRECT_DST := $(DST)/convDirect

OPENFHE_VER := aa391988d354d4360f390f223a90e0d1b98839d7
OPENFHE_SRC := $(SRC)/openfhe
OPENFHE_DST := $(DST)/openfhe

OPENFHE_PYTHON_VER := 59312e0eb490ffe9dc200e8426df72a533a1542a
OPENFHE_PYTHON_SRC := $(SRC)/openfhe-python
OPENFHE_PYTHON_DST := $(DST)/openfhe-python

UARCHFHE_VER := 7970d0dfad5b74939da492cf61c5d9c4a9753c19
UARCHFHE_SRC := $(SRC)/uarchfhe
UARCHFHE_DST := $(DST)/uarchfhe

POLYHE_VER := 804e4d48fbf792cfe61c1a8f283ed6ef1e907e7f
POLYHE_SRC := $(SRC)/polyhe
POLYHE_DST := $(DST)/polyhe

PYMPI_VER := a1bddf15ce0675a7748ec4ebf9fa2c779e0c9285
PYMPI_SRC := $(SRC)/pympi
PYMPI_DST := $(DST)/pympi

PYDTNN_VER := 
PYDTNN_SRC := $(CURDIR)
PYDTNN_DST := $(DST)/pydtnn

# ============================================================================
# Global targets
# ============================================================================

.PHONY: all deps src build install clean config

all: install

deps: \
	blis-deps \
	tvm-deps \
	convgemm-deps \
	convwinograd-deps \
	convdirect-deps \
	openfhe-deps \
	openfhe-python-deps \
	uarchfhe-deps \
	polyhe-deps \
	pympi-deps \
	pydtnn-deps

src: \
	blis-src \
	tvm-src \
	convgemm-src \
	convwinograd-src \
	convdirect-src \
	openfhe-src \
	openfhe-python-src \
	uarchfhe-src \
	polyhe-src \
	pympi-src \
	pydtnn-src

build: \
	blis-build \
	tvm-build \
	convgemm-build \
	convwinograd-build \
	convdirect-build \
	openfhe-build \
	openfhe-python-build \
	uarchfhe-build \
	polyhe-build \
	pympi-build \
	pydtnn-build

install: \
	blis-install \
	tvm-install \
	convgemm-install \
	convwinograd-install \
	convdirect-install \
	openfhe-install \
	openfhe-python-install \
	uarchfhe-install \
	polyhe-install \
	pympi-install \
	pydtnn-install

clean: \
	blis-clean \
	tvm-clean \
	convgemm-clean \
	convwinograd-clean \
	convdirect-clean \
	openfhe-clean \
	openfhe-python-clean \
	uarchfhe-clean \
	polyhe-clean \
	pympi-clean \
	pydtnn-clean

define ld_add
	[[ ":$${LD_LIBRARY_PATH}:" = *":$(1):"* ]] || LD_LIBRARY_PATH="$${LD_LIBRARY_PATH:+"$${LD_LIBRARY_PATH:?}:"}$(1)"
endef

config:
	@ \
	$(call ld_add,$(BLIS_DST)/lib); \
	$(call ld_add,$(TVM_DST)/lib); \
	$(call ld_add,$(CONVGEMM_DST)/lib); \
	$(call ld_add,$(CONVWINOGRAD_DST)/lib); \
	$(call ld_add,$(CONVDIRECT_DST)/lib); \
	$(call ld_add,$(OPENFHE_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

# ============================================================================
# BLIS
# ============================================================================

.PHONY: \
	blis \
	blis-deps \
	blis-src \
	blis-build \
	blis-install \
	blis-clean

blis: blis-install

blis-deps:
	apt-get install -y make gcc

blis-src: $(BLIS_SRC)/.git
$(BLIS_SRC)/.git:
	git submodule update --init "$(BLIS_SRC)"
	cd "$(BLIS_SRC)" && \
		git checkout "$(BLIS_VER)"

blis-build: $(BLIS_DST)/.build
$(BLIS_DST)/.build: $(BLIS_SRC)/.git
	mkdir -p "$(BLIS_DST)"
	cd "$(BLIS_SRC)" && \
		./configure \
			--prefix="$(BLIS_DST)" \
			--enable-cblas \
			auto && \
		make -j "$(PROCS)"
	cd "$(BLIS_SRC)" && \
		make install
	touch "$(BLIS_DST)/.build"

blis-install: $(BLIS_DST)/.build
	@ \
	$(call ld_add,$(BLIS_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

blis-clean:
	cd "$(BLIS_SRC)" && \
		rm -rf "$(BLIS_DST)" && \
		make clean

# ============================================================================
# TVM
# ============================================================================

.PHONY: \
	tvm \
	tvm-deps \
	tvm-src \
	tvm-build \
	tvm-install \
	tvm-clean

tvm: tvm-install

tvm-deps:
	apt-get install -y python3 cmake gcc llvm-dev
	pip install numpy psutil build

tvm-src: $(TVM_SRC)/.git
$(TVM_SRC)/.git:
	git submodule update --init --recursive "$(TVM_SRC)"
	cd "$(TVM_SRC)" && \
		git checkout "$(TVM_VER)"

tvm-build: $(TVM_DST)/.build
$(TVM_DST)/.build: $(TVM_SRC)/.git
	mkdir -p "$(TVM_DST)"
	cd "$(TVM_SRC)" && \
		mkdir -p build && \
		cd build && \
		cp ../cmake/config.cmake . && \
		cmake \
			-D CMAKE_INSTALL_PREFIX="$(TVM_DST)" \
			.. && \
		cmake --build . --parallel "$(PROCS)" && \
	cd "$(TVM_SRC)" && \
		cmake --install build && \
		python3 -m build -wo "$(TVM_DST)" ./3rdparty/tvm-ffi && \
		python3 -m build -wo "$(TVM_DST)" ./python
	touch "$(TVM_DST)/.build"

tvm-install: $(TVM_DST)/.build
	@ \
	$(call ld_add,$(TVM_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}
	pip install "$(TVM_DST)"/*.whl

tvm-clean:
	cd "$(TVM_SRC)" && \
		rm -rf "$(TVM_DST)" && \
		rm -rf "$(TVM_SRC)/build"

# ============================================================================
# convGemm
# ============================================================================

.PHONY: \
	convgemm \
	convgemm-deps \
	convgemm-src \
	convgemm-build \
	convgemm-install \
	convgemm-clean

convgemm: convgemm-install

convgemm-deps: # blis-install
	apt-get install -y cmake gcc

convgemm-src: $(CONVGEMM_SRC)/.git
$(CONVGEMM_SRC)/.git:
	git submodule update --init "$(CONVGEMM_SRC)"
	cd "$(CONVGEMM_SRC)" && \
		git checkout "$(CONVGEMM_VER)"

convgemm-build: $(CONVGEMM_DST)/.build
$(CONVGEMM_DST)/.build: $(CONVGEMM_SRC)/.git
	mkdir -p "$(CONVGEMM_DST)" "$(CONVGEMM_SRC)/build"
	cd "$(CONVGEMM_SRC)" && \
		cd build && \
		cmake \
			-D CMAKE_DST_PATH="$(BLIS_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(CONVGEMM_DST)" \
			.. && \
		cmake --build . --parallel "$(PROCS)"
	cd "$(CONVGEMM_SRC)" && \
		cmake --install build
	touch "$(CONVGEMM_DST)/.build"

convgemm-install: $(CONVGEMM_DST)/.build
	@ \
	$(call ld_add,$(CONVGEMM_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

convgemm-clean:
	cd "$(CONVGEMM_SRC)" && \
		rm -rf "$(CONVGEMM_DST)" && \
		rm -rf "$(CONVGEMM_SRC)/build"

# ============================================================================
# convWinograd
# ============================================================================

.PHONY: \
	convwinograd \
	convwinograd-deps \
	convwinograd-src \
	convwinograd-build \
	convwinograd-install \
	convwinograd-clean

convwinograd: convwinograd-install

convwinograd-deps: # blis-install
	apt-get install -y cmake gcc

convwinograd-src: $(CONVWINOGRAD_SRC)/.git
$(CONVWINOGRAD_SRC)/.git:
	git submodule update --init "$(CONVWINOGRAD_SRC)"
	cd "$(CONVWINOGRAD_SRC)" && \
		git checkout "$(CONVWINOGRAD_VER)"

convwinograd-build: $(CONVWINOGRAD_DST)/.build
$(CONVWINOGRAD_DST)/.build: $(CONVWINOGRAD_SRC)/.git
	mkdir -p "$(CONVWINOGRAD_DST)" "$(CONVWINOGRAD_SRC)/build"
	cd "$(CONVWINOGRAD_SRC)" && \
		cd build && \
		cmake \
			-D BLA_VENDOR=FLAME \
			-D CMAKE_PREFIX_PATH="$(BLIS_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(CONVWINOGRAD_DST)" \
			.. && \
		cmake --build . --parallel "$(PROCS)"
	cd "$(CONVWINOGRAD_SRC)" && \
		cmake --install build
	touch "$(CONVWINOGRAD_DST)/.build"

convwinograd-install: $(CONVWINOGRAD_DST)/.build
	@ \
	$(call ld_add,$(CONVWINOGRAD_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

convwinograd-clean:
	cd "$(CONVWINOGRAD_SRC)" && \
		rm -rf "$(CONVWINOGRAD_DST)" && \
		rm -rf "$(CONVWINOGRAD_SRC)/build"

# ============================================================================
# convDirect
# ============================================================================

.PHONY: \
	convdirect \
	convdirect-deps \
	convdirect-src \
	convdirect-build \
	convdirect-install \
	convdirect-clean

convdirect: convdirect-install

convdirect-deps: # blis-install tvm-install convgemm-install
	apt-get install -y cmake gcc

convdirect-src: $(CONVDIRECT_SRC)/.git
$(CONVDIRECT_SRC)/.git:
	git submodule update --init --recursive "$(CONVDIRECT_SRC)"
	cd "$(CONVDIRECT_SRC)" && \
		git checkout "$(CONVDIRECT_VER)"

convdirect-build: $(CONVDIRECT_DST)/.build
$(CONVDIRECT_DST)/.build: $(CONVDIRECT_SRC)/.git
	mkdir -p "$(CONVDIRECT_DST)" "$(CONVDIRECT_SRC)/build"
	cd "$(CONVDIRECT_SRC)" && \
		cd build && \
		cmake \
			-D CMAKE_DST_PATH="$(BLIS_DST);$(TVM_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(CONVDIRECT_DST)" \
			.. && \
		cmake --build . --parallel "$(PROCS)"
	cd "$(CONVDIRECT_SRC)" && \
		cmake --install build
	touch "$(CONVDIRECT_DST)/.build"

convdirect-install: $(CONVDIRECT_DST)/.build
	@ \
	$(call ld_add,$(CONVDIRECT_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

convdirect-clean:
	cd "$(CONVDIRECT_SRC)" && \
		rm -rf "$(CONVDIRECT_DST)" && \
		rm -rf "$(CONVDIRECT_SRC)/build"

# ============================================================================
# OpenFHE
# ============================================================================

.PHONY: \
	openfhe \
	openfhe-deps \
	openfhe-src \
	openfhe-build \
	openfhe-install \
	openfhe-clean

openfhe: openfhe-install

openfhe-deps:
	apt-get install -y cmake gcc

openfhe-src: $(OPENFHE_SRC)/.git
$(OPENFHE_SRC)/.git:
	git submodule update --init --recursive "$(OPENFHE_SRC)"
	cd "$(OPENFHE_SRC)" && \
		git checkout "$(OPENFHE_VER)"

openfhe-build: $(OPENFHE_DST)/.build
$(OPENFHE_DST)/.build: $(OPENFHE_SRC)/.git
	mkdir -p "$(OPENFHE_DST)" "$(OPENFHE_SRC)/build"
	cd "$(OPENFHE_SRC)" && \
		cd build && \
		cmake \
			-D CMAKE_INSTALL_PREFIX="$(OPENFHE_DST)" \
			.. && \
		cmake --build . --parallel "$(PROCS)"
	cd "$(OPENFHE_SRC)" && \
		cmake --install build
	touch "$(OPENFHE_DST)/.build"

openfhe-install: $(OPENFHE_DST)/.build
	@ \
	$(call ld_add,$(OPENFHE_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

openfhe-clean:
	cd "$(OPENFHE_SRC)" && \
		rm -rf "$(OPENFHE_DST)" && \
		rm -rf "$(OPENFHE_SRC)/build"

# ============================================================================
# OpenFHE Python
# ============================================================================

.PHONY: \
	openfhe-python \
	openfhe-python-deps \
	openfhe-python-src \
	openfhe-python-build \
	openfhe-python-install \
	openfhe-python-clean

openfhe-python: openfhe-python-install

openfhe-python-deps: # openfhe-install
	apt-get install -y python3 cmake gcc
	pip install pybind11[global]

openfhe-python-src: $(OPENFHE_PYTHON_SRC)/.git
$(OPENFHE_PYTHON_SRC)/.git:
	git submodule update --init "$(OPENFHE_PYTHON_SRC)"
	cd "$(OPENFHE_PYTHON_SRC)" && \
		git checkout "$(OPENFHE_PYTHON_VER)"

openfhe-python-build: $(OPENFHE_PYTHON_DST)/.build
$(OPENFHE_PYTHON_DST)/.build: $(OPENFHE_PYTHON_SRC)/.git
	mkdir -p "$(OPENFHE_PYTHON_DST)" "$(OPENFHE_PYTHON_SRC)/build"
	cd "$(OPENFHE_PYTHON_SRC)" && \
		cd build && \
		cmake \
			-D CMAKE_PREFIX_PATH="$(OPENFHE_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(OPENFHE_PYTHON_SRC)/openfhe" \
			.. && \
		cmake --build . --parallel "$(PROCS)" && \
	cd "$(OPENFHE_PYTHON_SRC)" && \
		cmake --install build && \
		printf '%s\n' > "$(OPENFHE_PYTHON_SRC)/pyproject.toml" \
			'[project]' \
			'name = "openfhe"' \
			'version = "1.4.2"' \
			'[build-system]' \
			'requires = ["setuptools"]' \
			'build-backend = "setuptools.build_meta"' \
			'[tool.setuptools]' \
			'packages.find.include = ["openfhe", "openfhe.*"]' \
			'package-data.openfhe = ["*"]' && \
		python3 -m build -wo "$(OPENFHE_PYTHON_DST)"
	touch "$(OPENFHE_PYTHON_DST)/.build"

openfhe-python-install: $(OPENFHE_PYTHON_DST)/.build
	pip install "$(OPENFHE_PYTHON_DST)"/*.whl

openfhe-python-clean:
	cd "$(OPENFHE_PYTHON_SRC)" && \
		rm -rf "$(OPENFHE_PYTHON_DST)" && \
		rm -rf "$(OPENFHE_PYTHON_SRC)/build" "$(OPENFHE_PYTHON_SRC)/pyproject.toml" "$(OPENFHE_PYTHON_SRC)/openfhe"

# ============================================================================
# uArchFHE
# ============================================================================

.PHONY: \
	uarchfhe \
	uarchfhe-deps \
	uarchfhe-src \
	uarchfhe-build \
	uarchfhe-install \
	uarchfhe-clean

uarchfhe: uarchfhe-install

uarchfhe-deps:
	apt-get install -y python3 libgmp-dev libntl-dev libbz2-dev
	pip install build

uarchfhe-src: $(UARCHFHE_SRC)/.git
$(UARCHFHE_SRC)/.git:
	git submodule update --init "$(UARCHFHE_SRC)"
	cd "$(UARCHFHE_SRC)" && \
		git checkout "$(UARCHFHE_VER)"

uarchfhe-build: $(UARCHFHE_DST)/.build
$(UARCHFHE_DST)/.build: $(UARCHFHE_SRC)/.git
	mkdir -p "$(UARCHFHE_DST)"
	cd "$(UARCHFHE_SRC)/crates/fhe_py_binding" && \
		python3 -m build -wo "$(UARCHFHE_DST)"
	touch "$(UARCHFHE_DST)/.build"

uarchfhe-install: $(UARCHFHE_DST)/.build
	pip install "$(UARCHFHE_DST)"/*.whl

uarchfhe-clean:
	cd "$(UARCHFHE_SRC)" && \
		rm -rf "$(UARCHFHE_DST)"

# ============================================================================
# PolyHE
# ============================================================================

.PHONY: \
	polyhe \
	polyhe-deps \
	polyhe-src \
	polyhe-build \
	polyhe-install \
	polyhe-clean

polyhe: polyhe-install

polyhe-deps:
	apt-get install -y python3
	pip install build

polyhe-src: $(POLYHE_SRC)/.git
$(POLYHE_SRC)/.git:
	git submodule update --init "$(POLYHE_SRC)"
	cd "$(POLYHE_SRC)" && \
		git checkout "$(POLYHE_VER)"

polyhe-build: $(POLYHE_DST)/.build
$(POLYHE_DST)/.build: $(POLYHE_SRC)/.git
	mkdir -p "$(POLYHE_DST)"
	cd "$(POLYHE_SRC)" && \
		python3 -m build -wo "$(POLYHE_DST)"
	touch "$(POLYHE_DST)/.build"

polyhe-install: $(POLYHE_DST)/.build
	pip install "$(POLYHE_DST)"/*.whl

polyhe-clean:
	cd "$(POLYHE_SRC)" && \
		rm -rf "$(POLYHE_DST)"

# ============================================================================
# PyMPI
# ============================================================================

.PHONY: \
	pympi \
	pympi-deps \
	pympi-src \
	pympi-build \
	pympi-install \
	pympi-clean

pympi: pympi-install

pympi-deps:
	apt-get install -y python3
	pip install build

pympi-src: $(PYMPI_SRC)/.git
$(PYMPI_SRC)/.git:
	git submodule update --init "$(PYMPI_SRC)"
	cd "$(PYMPI_SRC)" && \
		git checkout "$(PYMPI_VER)"

pympi-build: $(PYMPI_DST)/.build
$(PYMPI_DST)/.build: $(PYMPI_SRC)/.git
	mkdir -p "$(PYMPI_DST)"
	cd "$(PYMPI_SRC)" && \
		python3 -m build -wo "$(PYMPI_DST)"
	touch "$(PYMPI_DST)/.build"

pympi-install: $(PYMPI_DST)/.build
	pip install "$(PYMPI_DST)"/*.whl

pympi-clean:
	cd "$(PYMPI_SRC)" && \
		rm -rf "$(PYMPI_DST)"

# ============================================================================
# PyDTNN
# ============================================================================

.PHONY: \
	pydtnn \
	pydtnn-deps \
	pydtnn-src \
	pydtnn-build \
	pydtnn-install \
	pydtnn-develop \
	pydtnn-clean

pydtnn: pydtnn-install

pydtnn-deps:
	apt-get install -y python3 gcc patchelf
	pip install build auditwheel

pydtnn-src: $(PYDTNN_SRC)/.git
$(PYDTNN_SRC)/.git:
	cd "$(PYDTNN_SRC)" && \
		git checkout "$(PYDTNN_VER)"

pydtnn-build: $(PYDTNN_DST)/.build
$(PYDTNN_DST)/.build: $(PYDTNN_SRC)/.git
	mkdir -p "$(PYDTNN_DST)" "$(PYDTNN_SRC)/build"
	cd "$(PYDTNN_SRC)" && \
		python3 -m build -so "$(PYDTNN_DST)" && \
		python3 -m build -wo "$(PYDTNN_SRC)/build" && \
		python3 -m auditwheel repair -w "$(PYDTNN_DST)" "$(PYDTNN_SRC)/build"/pydtnn-*.whl && \
	touch "$(PYDTNN_DST)/.build"

pydtnn-install: $(PYDTNN_DST)/.build
	pip install "$(PYDTNN_DST)"/pydtnn-*.whl

pydtnn-develop:
	pip install --config-settings editable_mode=compat -e "$(PYDTNN_SRC)"

pydtnn-clean:
	cd "$(PYDTNN_SRC)" && \
		rm -rf "$(PYDTNN_DST)" && \
		rm -rf "$(PYDTNN_SRC)/build"/pydtnn-*.whl
