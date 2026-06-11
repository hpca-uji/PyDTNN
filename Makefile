# ============================================================================
# Configuration
# ============================================================================

SHELL := $(shell which bash)
PROCS := $(shell nproc)

APT := $(shell which apt-get)
PIP := $(shell which pip3)

SRC := $(CURDIR)/vendor
DST := $(CURDIR)/build

define GIT_VER
$(shell git rev-parse "HEAD:$$(realpath -m --relative-base "$(CURDIR)" "$(1)" | grep -vxF ".")")
endef

BLIS_SRC := $(SRC)/blis
BLIS_DST := $(DST)/blis
BLIS_VER := b04de636c1702e4cb8e7ad82bab3cf43d2dbdfc6

TVM_SRC := $(SRC)/tvm
TVM_DST := $(DST)/tvm
TVM_VER := 43e9c275b6e85d7631e54c8468b49b4706cd674a

CONVGEMM_SRC := $(SRC)/convGemm
CONVGEMM_DST := $(DST)/convGemm
CONVGEMM_VER := 1ebea3c77cd961cb207f5964025733913765b0e6

CONVWINOGRAD_SRC := $(SRC)/convWinograd
CONVWINOGRAD_DST := $(DST)/convWinograd
CONVWINOGRAD_VER := fc2d5af8d0ee551e508b97082ee7aab3bbff0244

CONVDIRECT_SRC := $(SRC)/convDirect
CONVDIRECT_DST := $(DST)/convDirect
CONVDIRECT_VER := 25937a6b3e06cf06089e7403798547c31528cba3

OPENFHE_SRC := $(SRC)/openfhe
OPENFHE_DST := $(DST)/openfhe
OPENFHE_VER := aa391988d354d4360f390f223a90e0d1b98839d7

OPENFHE_PYTHON_SRC := $(SRC)/openfhe-python
OPENFHE_PYTHON_DST := $(DST)/openfhe-python
OPENFHE_PYTHON_VER := 59312e0eb490ffe9dc200e8426df72a533a1542a

UARCHFHE_SRC := $(SRC)/uarchfhe
UARCHFHE_DST := $(DST)/uarchfhe
UARCHFHE_VER := bca9858f96767f204b3ce9ed577f056ec88f95e1

PYDTNN_SRC := $(CURDIR)
PYDTNN_DST := $(DST)/pydtnn
PYDTNN_VER := $(call GIT_VER,$(PYDTNN_SRC))

# ============================================================================
# Global
# ============================================================================

.PHONY: \
	help \
	deps \
	src \
	build \
	install \
	format \
	check \
	clean \
	env

.DEFAULT_GOAL := pydtnn-develop

deps: pydtnn-deps
src: pydtnn-src
build: pydtnn-build
install: pydtnn-install
format: pydtnn-format
check: pydtnn-check
clean: pydtnn-clean

help:
	@echo PyDTNN Makefile
	@echo
	@echo Targets:
	@printf -- '- %s\n' \
		deps \
		src \
		build \
		install \
		clean
	@echo
	@echo Packages:
	@printf -- '- %s\n' \
		'blis ($(BLIS_VER))' \
		'tvm ($(TVM_VER))' \
		'convgemm ($(CONVGEMM_VER))' \
		'convwinograd ($(CONVWINOGRAD_VER))' \
		'convdirect ($(CONVDIRECT_VER))' \
		'openfhe ($(OPENFHE_VER))' \
		'openfhe-python ($(OPENFHE_PYTHON_VER))' \
		'uarchfhe ($(UARCHFHE_VER))' \
		'pydtnn ($(PYDTNN_VER))'
	@echo
	@echo Special:
	@printf -- '- %s\n' \
		env \
		pydtnn-{check,format} \
		'$${package}-$${target}'

define LD_ADD
[[ ":$${LD_LIBRARY_PATH}:" = *":$(1):"* ]] \
|| LD_LIBRARY_PATH="$${LD_LIBRARY_PATH:+"$${LD_LIBRARY_PATH:?}:"}$(1)"
endef

env:
	@ \
	$(call LD_ADD,$(BLIS_DST)/lib); \
	$(call LD_ADD,$(TVM_DST)/lib); \
	$(call LD_ADD,$(CONVGEMM_DST)/lib); \
	$(call LD_ADD,$(CONVWINOGRAD_DST)/lib); \
	$(call LD_ADD,$(CONVDIRECT_DST)/lib); \
	$(call LD_ADD,$(OPENFHE_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

$(DST)/.gitignore:
	mkdir -p "$(DST)"
	echo "*" > "$@"

# ============================================================================
# Vendor
# ============================================================================

.PHONY: \
	vendor \
	vendor-deps \
	vendor-src \
	vendor-build \
	vendor-install \
	vendor-clean

vendor: vendor-build

vendor-deps: \
	blis-deps \
	tvm-deps \
	convgemm-deps \
	convwinograd-deps \
	convdirect-deps \
	openfhe-deps \
	openfhe-python-deps
#	uarchfhe-deps

vendor-src: \
	blis-src \
	tvm-src \
	convgemm-src \
	convwinograd-src \
	convdirect-src \
	openfhe-src \
	openfhe-python-src
#	uarchfhe-src

vendor-build: \
	blis-build \
	tvm-build \
	convgemm-build \
	convwinograd-build \
	convdirect-build \
	openfhe-build \
	openfhe-python-build
#	uarchfhe-build

vendor-install: \
	blis-install \
	tvm-install \
	convgemm-install \
	convwinograd-install \
	convdirect-install \
	openfhe-install \
	openfhe-python-install
#	uarchfhe-install

vendor-clean: \
	blis-clean \
	tvm-clean \
	convgemm-clean \
	convwinograd-clean \
	convdirect-clean \
	openfhe-clean \
	openfhe-python-clean
#	uarchfhe-clean

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

blis: blis-build

blis-deps:
	$(APT) install -y make gcc

blis-src: $(BLIS_SRC)/.git
$(BLIS_SRC)/.git:
	git submodule update --init "$(BLIS_SRC)"
	cd "$(BLIS_SRC)" && \
		git checkout "$(BLIS_VER)"

blis-build: $(BLIS_DST)/.build
$(BLIS_DST)/.build: | $(BLIS_SRC)/.git $(DST)/.gitignore
	mkdir -p "$(BLIS_DST)"
	cd "$(BLIS_SRC)" && \
		./configure \
			--prefix="$(BLIS_DST)" \
			--enable-cblas \
			auto && \
		make -j "$(PROCS)"
	cd "$(BLIS_SRC)" && \
		make install
	echo "$(BLIS_VER)" > "$@"

blis-install: $(BLIS_DST)/.build
	@ \
	$(call LD_ADD,$(BLIS_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

blis-clean:
	cd "$(BLIS_SRC)" && \
		rm -rf "$(BLIS_DST)" && \
		(make clean || true)

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

tvm: tvm-build

tvm-deps:
	$(APT) install -y python3 cmake gcc llvm-dev zlib1g-dev libxml2-dev
	$(PIP) install numpy psutil build

tvm-src: $(TVM_SRC)/.git
$(TVM_SRC)/.git:
	git submodule update --init --recursive "$(TVM_SRC)"
	cd "$(TVM_SRC)" && \
		git checkout "$(TVM_VER)"

tvm-build: $(TVM_DST)/.build
$(TVM_DST)/.build: | $(TVM_SRC)/.git $(DST)/.gitignore
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
	echo "$(TVM_VER)" > "$@"

tvm-install: $(TVM_DST)/.build
	@ \
	$(call LD_ADD,$(TVM_DST)/lib); \
	echo export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}
	$(PIP) install "$(TVM_DST)"/*.whl

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

convgemm: convgemm-build

convgemm-deps:
	@echo make blis-install
	$(APT) install -y cmake gcc

convgemm-src: $(CONVGEMM_SRC)/.git
$(CONVGEMM_SRC)/.git:
	git submodule update --init "$(CONVGEMM_SRC)"
	cd "$(CONVGEMM_SRC)" && \
		git checkout "$(CONVGEMM_VER)"

convgemm-build: $(CONVGEMM_DST)/.build
$(CONVGEMM_DST)/.build: | $(CONVGEMM_SRC)/.git $(DST)/.gitignore
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
	echo "$(CONVGEMM_VER)" > "$@"

convgemm-install: $(CONVGEMM_DST)/.build
	@ \
	$(call LD_ADD,$(CONVGEMM_DST)/lib); \
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

convwinograd: convwinograd-build

convwinograd-deps:
	@echo make blis-install
	$(APT) install -y cmake gcc

convwinograd-src: $(CONVWINOGRAD_SRC)/.git
$(CONVWINOGRAD_SRC)/.git:
	git submodule update --init "$(CONVWINOGRAD_SRC)"
	cd "$(CONVWINOGRAD_SRC)" && \
		git checkout "$(CONVWINOGRAD_VER)"

convwinograd-build: $(CONVWINOGRAD_DST)/.build
$(CONVWINOGRAD_DST)/.build: | $(CONVWINOGRAD_SRC)/.git $(DST)/.gitignore
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
	echo "$(CONVWINOGRAD_VER)" > "$@"

convwinograd-install: $(CONVWINOGRAD_DST)/.build
	@ \
	$(call LD_ADD,$(CONVWINOGRAD_DST)/lib); \
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

convdirect: convdirect-build

convdirect-deps:
	@echo make blis-install tvm-install convgemm-install
	$(APT) install -y cmake gcc

convdirect-src: $(CONVDIRECT_SRC)/.git
$(CONVDIRECT_SRC)/.git:
	git submodule update --init --recursive "$(CONVDIRECT_SRC)"
	cd "$(CONVDIRECT_SRC)" && \
		git checkout "$(CONVDIRECT_VER)"

convdirect-build: $(CONVDIRECT_DST)/.build
$(CONVDIRECT_DST)/.build: | $(CONVDIRECT_SRC)/.git $(DST)/.gitignore
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
	echo "$(CONVDIRECT_VER)" > "$@"

convdirect-install: $(CONVDIRECT_DST)/.build
	@ \
	$(call LD_ADD,$(CONVDIRECT_DST)/lib); \
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

openfhe: openfhe-build

openfhe-deps:
	$(APT) install -y cmake gcc

openfhe-src: $(OPENFHE_SRC)/.git
$(OPENFHE_SRC)/.git:
	git submodule update --init --recursive "$(OPENFHE_SRC)"
	cd "$(OPENFHE_SRC)" && \
		git checkout "$(OPENFHE_VER)"

openfhe-build: $(OPENFHE_DST)/.build
$(OPENFHE_DST)/.build: | $(OPENFHE_SRC)/.git $(DST)/.gitignore
	mkdir -p "$(OPENFHE_DST)" "$(OPENFHE_SRC)/build"
	cd "$(OPENFHE_SRC)" && \
		cd build && \
		cmake \
			-D CMAKE_INSTALL_PREFIX="$(OPENFHE_DST)" \
			.. && \
		cmake --build . --parallel "$(PROCS)"
	cd "$(OPENFHE_SRC)" && \
		cmake --install build
	echo "$(OPENFHE_VER)" > "$@"

openfhe-install: $(OPENFHE_DST)/.build
	@ \
	$(call LD_ADD,$(OPENFHE_DST)/lib); \
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

openfhe-python: openfhe-python-build

openfhe-python-deps:
	@echo make openfhe-install
	$(APT) install -y python3 cmake gcc
	$(PIP) install pybind11[global]

openfhe-python-src: $(OPENFHE_PYTHON_SRC)/.git
$(OPENFHE_PYTHON_SRC)/.git:
	git submodule update --init "$(OPENFHE_PYTHON_SRC)"
	cd "$(OPENFHE_PYTHON_SRC)" && \
		git checkout "$(OPENFHE_PYTHON_VER)"

openfhe-python-build: $(OPENFHE_PYTHON_DST)/.build
$(OPENFHE_PYTHON_DST)/.build: | $(OPENFHE_PYTHON_SRC)/.git $(DST)/.gitignore
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
	echo "$(OPENFHE_PYTHON_VER)" > "$@"

openfhe-python-install: $(OPENFHE_PYTHON_DST)/.build
	$(PIP) install "$(OPENFHE_PYTHON_DST)"/*.whl

openfhe-python-clean:
	cd "$(OPENFHE_PYTHON_SRC)" && \
		rm -rf "$(OPENFHE_PYTHON_DST)" && \
		rm -rf \
			"$(OPENFHE_PYTHON_SRC)/build" \
			"$(OPENFHE_PYTHON_SRC)/pyproject.toml" \
			"$(OPENFHE_PYTHON_SRC)/openfhe"

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

uarchfhe: uarchfhe-build

uarchfhe-deps:
	$(APT) install -y python3 libgmp-dev libntl-dev libbz2-dev
	$(PIP) install build

uarchfhe-src: $(UARCHFHE_SRC)/.git
$(UARCHFHE_SRC)/.git:
	git submodule update --init "$(UARCHFHE_SRC)"
	cd "$(UARCHFHE_SRC)" && \
		git checkout "$(UARCHFHE_VER)"

uarchfhe-build: $(UARCHFHE_DST)/.build
$(UARCHFHE_DST)/.build: | $(UARCHFHE_SRC)/.git $(DST)/.gitignore
	mkdir -p "$(UARCHFHE_DST)"
	cd "$(UARCHFHE_SRC)/crates/fhe_py_binding" && \
		python3 -m build -wo "$(UARCHFHE_DST)"
	echo "$(UARCHFHE_VER)" > "$@"

uarchfhe-install: $(UARCHFHE_DST)/.build
	$(PIP) install "$(UARCHFHE_DST)"/*.whl

uarchfhe-clean:
	cd "$(UARCHFHE_SRC)" && \
		rm -rf "$(UARCHFHE_DST)"

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
	pydtnn-format \
	pydtnn-check \
	pydtnn-clean

pydtnn: pydtnn-build

pydtnn-deps:
	$(APT) install -y python3 gcc patchelf
	$(PIP) install build auditwheel

pydtnn-src: $(PYDTNN_SRC)/.git
$(PYDTNN_SRC)/.git:
	cd "$(PYDTNN_SRC)" && \
		git checkout "$(PYDTNN_VER)"

pydtnn-build: $(PYDTNN_DST)/.build
$(PYDTNN_DST)/.build: | $(PYDTNN_SRC)/.git $(DST)/.gitignore
	mkdir -p "$(PYDTNN_DST)" "$(PYDTNN_SRC)/build"
	cd "$(PYDTNN_SRC)" && \
		python3 -m build -so "$(PYDTNN_DST)" && \
		python3 -m build -wo "$(PYDTNN_SRC)/build" && \
		python3 -m auditwheel repair -w "$(PYDTNN_DST)" "$(PYDTNN_SRC)/build"/pydtnn-*.whl && \
	echo "$(PYDTNN_VER)" > "$@"

pydtnn-install: $(PYDTNN_DST)/.build
	$(PIP) install "$(PYDTNN_DST)"/pydtnn-*.whl

pydtnn-develop:
	$(PIP) install \
		--config-settings editable_mode=compat \
		-e "$(PYDTNN_SRC)"[dev]

pydtnn-format:
	"$(PYDTNN_SRC)/scripts/srcs/format.sh" pydtnn
pydtnn-check:
	# --dist=loadscope
	mkdir -p "$(PYDTNN_SRC)/build/.tmp"
	cd "$(PYDTNN_SRC)/build/.tmp" && \
		pytest -v -n "$(PROCS)" \
			--junitxml="$(PYDTNN_SRC)/build/tests" \
			--cov --cov-config="$(PYDTNN_SRC)/pyproject.toml" \
			--cov-report=term --cov-report=xml:"$(PYDTNN_SRC)/build/coverage" \
			--pyargs pydtnn.tests.groups.all && \
		flake8 \
			--exit-zero \
			--format=gl-codeclimate \
			--output-file="$(PYDTNN_SRC)/build/quality"

pydtnn-clean:
	cd "$(PYDTNN_SRC)" && \
		rm -rf "$(PYDTNN_DST)" && \
		rm -rf "$(PYDTNN_SRC)/build"/pydtnn-*.whl && \
		find "$(PYDTNN_SRC)/pydtnn" \
			-iname "*.pyc" \
			-iname "*.so" \
			-iname "*.dll" \
			-iname "*.dsym" \
			-delete
