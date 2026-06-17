# ============================================================================
# Configuration
# ============================================================================

SHELL := bash
NPROC := $(shell nproc)

APT := sudo apt-get
PIP := pip3

SRC := $(CURDIR)/vendor
DST := $(CURDIR)/build

BLIS_SRC := $(SRC)/blis
BLIS_DST := $(DST)/blis

TVM_SRC := $(SRC)/tvm
TVM_DST := $(DST)/tvm

CONVGEMM_SRC := $(SRC)/convGemm
CONVGEMM_DST := $(DST)/convGemm

CONVWINOGRAD_SRC := $(SRC)/convWinograd
CONVWINOGRAD_DST := $(DST)/convWinograd

CONVDIRECT_SRC := $(SRC)/convDirect
CONVDIRECT_DST := $(DST)/convDirect

OPENFHE_SRC := $(SRC)/openfhe
OPENFHE_DST := $(DST)/openfhe

OPENFHE_PYTHON_SRC := $(SRC)/openfhe-python
OPENFHE_PYTHON_DST := $(DST)/openfhe-python

UARCHFHE_SRC := $(SRC)/uarchfhe
UARCHFHE_DST := $(DST)/uarchfhe

PYDTNN_SRC := $(CURDIR)
PYDTNN_DST := $(DST)/pydtnn

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
	test \
	lint \
	clean \
	sync \
	env

.DEFAULT_GOAL := pydtnn-develop

help:
	@echo "PyDTNN's Makefile"
	@echo
	@echo Targets:
	@printf -- '- %s\n' \
		deps \
		src \
		build \
		install \
		format \
		test \
		lint \
		clean
	@echo
	@echo Packages:
	@printf -- '- %s\n' \
		blis \
		tvm \
		convgemm \
		convwinograd \
		convdirect \
		openfhe \
		openfhe-python \
		pydtnn
	@echo
	@echo Special:
	@printf -- '- %s\n' \
		sync \
		env

deps: \
	pydtnn-deps \
	blis-deps \
	tvm-deps \
	convgemm-deps \
	convwinograd-deps \
	convdirect-deps \
	openfhe-deps \
	openfhe-python-deps

src: \
	pydtnn-src \
	blis-src \
	tvm-src \
	convgemm-src \
	convwinograd-src \
	convdirect-src \
	openfhe-src \
	openfhe-python-src

build: \
	$(DST)/.gitignore \
	blis-build \
	tvm-build \
	convgemm-build \
	convwinograd-build \
	convdirect-build \
	openfhe-build \
	openfhe-python-build \
	pydtnn-build \

install: \
	blis-install \
	tvm-install \
	convgemm-install \
	convwinograd-install \
	convdirect-install \
	openfhe-install \
	openfhe-python-install \
	pydtnn-install

format: \
	pydtnn-format

test: \
	pydtnn-test

lint: \
	pydtnn-lint

clean: \
	pydtnn-clean \
	openfhe-python-clean \
	openfhe-clean \
	convdirect-clean \
	convwinograd-clean \
	convgemm-clean \
	tvm-clean \
	blis-clean
	rm -rf "$(DST)"

define VER_SYNC
	[ ! -e "$(1)/.git" ] || { \
		COMMIT=$$(git -C "$(1)" rev-parse HEAD) && \
		DATE=$$(git -C "$(1)" show -s --format=%ci $${COMMIT:?}) && \
		touch -d "$${DATE:?}" "$(1)/.git"; }
	[ ! -e "$(2)/.build" ] || { \
		COMMIT=$$(cat "$(2)/.build") && \
		DATE=$$(git -C "$(1)" show -s --format=%ci $${COMMIT:?}) && \
		touch -d "$${DATE:?}" "$(2)/.build"; }
endef

sync:
	$(call VER_SYNC,$(BLIS_SRC),$(BLIS_DST))
	$(call VER_SYNC,$(TVM_SRC),$(TVM_DST))
	$(call VER_SYNC,$(CONVGEMM_SRC),$(CONVGEMM_DST))
	$(call VER_SYNC,$(CONVWINOGRAD_SRC),$(CONVWINOGRAD_DST))
	$(call VER_SYNC,$(CONVDIRECT_SRC),$(CONVDIRECT_DST))
	$(call VER_SYNC,$(OPENFHE_SRC),$(OPENFHE_DST))
	$(call VER_SYNC,$(OPENFHE_PYTHON_SRC),$(OPENFHE_PYTHON_DST))
	$(call VER_SYNC,$(PYDTNN_SRC),$(PYDTNN_DST))

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
	printf '%s\n' "LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}"

$(DST)/.gitignore:
	mkdir -p "$(DST)"
	echo "*" > "$@"

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
	git submodule update --init --recursive "$(BLIS_SRC)"

blis-build: $(BLIS_DST)/.build
$(BLIS_DST)/.build: $(BLIS_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(BLIS_DST)"
	cd "$(BLIS_SRC)" && \
		./configure \
			--prefix="$(BLIS_DST)" \
			--enable-cblas \
			auto && \
		make -j "$(NPROC)"
	cd "$(BLIS_SRC)" && \
		make install
	git -C "$(BLIS_SRC)" rev-parse HEAD > "$@"

blis-install: $(BLIS_DST)/.build
	$(call LD_ADD,$(BLIS_DST)/lib); \
	export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

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

tvm-build: $(TVM_DST)/.build
$(TVM_DST)/.build: $(TVM_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(TVM_DST)" "$(TVM_SRC)/build"
	cd "$(TVM_SRC)/build" && \
		cp ../cmake/config.cmake . && \
		cmake \
			-D CMAKE_C_FLAGS="-w" \
			-D CMAKE_CXX_FLAGS="-w" \
			-D CMAKE_INSTALL_PREFIX="$(TVM_DST)" \
			.. && \
		cmake --build . --parallel "$(NPROC)" && \
	cd "$(TVM_SRC)" && \
		cmake --install build && \
		python3 -m build -wo "$(TVM_DST)" ./3rdparty/tvm-ffi && \
		python3 -m build -wo "$(TVM_DST)" ./python
	git -C "$(TVM_SRC)" rev-parse HEAD > "$@"

tvm-install: $(TVM_DST)/.build
	@ \
	$(call LD_ADD,$(TVM_DST)/lib); \
	export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}
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
	git submodule update --init --recursive "$(CONVGEMM_SRC)"

convgemm-build: $(CONVGEMM_DST)/.build
$(CONVGEMM_DST)/.build: $(CONVGEMM_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(CONVGEMM_DST)" "$(CONVGEMM_SRC)/build"
	cd "$(CONVGEMM_SRC)/build" && \
		cmake \
			-D CMAKE_DST_PATH="$(BLIS_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(CONVGEMM_DST)" \
			.. && \
		cmake --build . --parallel "$(NPROC)"
	cd "$(CONVGEMM_SRC)" && \
		cmake --install build
	git -C "$(CONVGEMM_SRC)" rev-parse HEAD > "$@"

convgemm-install: $(CONVGEMM_DST)/.build
	@ \
	$(call LD_ADD,$(CONVGEMM_DST)/lib); \
	export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

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
	git submodule update --init --recursive "$(CONVWINOGRAD_SRC)"

convwinograd-build: $(CONVWINOGRAD_DST)/.build
$(CONVWINOGRAD_DST)/.build: $(CONVWINOGRAD_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(CONVWINOGRAD_DST)" "$(CONVWINOGRAD_SRC)/build"
	cd "$(CONVWINOGRAD_SRC)/build" && \
		cmake \
			-D BLA_VENDOR=FLAME \
			-D CMAKE_PREFIX_PATH="$(BLIS_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(CONVWINOGRAD_DST)" \
			.. && \
		cmake --build . --parallel "$(NPROC)"
	cd "$(CONVWINOGRAD_SRC)" && \
		cmake --install build
	git -C "$(CONVWINOGRAD_SRC)" rev-parse HEAD > "$@"

convwinograd-install: $(CONVWINOGRAD_DST)/.build
	@ \
	$(call LD_ADD,$(CONVWINOGRAD_DST)/lib); \
	export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

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

convdirect-build: $(CONVDIRECT_DST)/.build
$(CONVDIRECT_DST)/.build: $(CONVDIRECT_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(CONVDIRECT_DST)" "$(CONVDIRECT_SRC)/build"
	cd "$(CONVDIRECT_SRC)/build" && \
		cmake \
			-D CMAKE_DST_PATH="$(BLIS_DST);$(TVM_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(CONVDIRECT_DST)" \
			.. && \
		cmake --build . --parallel "$(NPROC)"
	cd "$(CONVDIRECT_SRC)" && \
		cmake --install build
	git -C "$(CONVDIRECT_SRC)" rev-parse HEAD > "$@"

convdirect-install: $(CONVDIRECT_DST)/.build
	@ \
	$(call LD_ADD,$(CONVDIRECT_DST)/lib); \
	export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

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

openfhe-build: $(OPENFHE_DST)/.build
$(OPENFHE_DST)/.build: $(OPENFHE_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(OPENFHE_DST)" "$(OPENFHE_SRC)/build"
	cd "$(OPENFHE_SRC)/build" && \
		cmake \
			-D CMAKE_INSTALL_PREFIX="$(OPENFHE_DST)" \
			.. && \
		cmake --build . --parallel "$(NPROC)"
	cd "$(OPENFHE_SRC)" && \
		cmake --install build
	git -C "$(OPENFHE_SRC)" rev-parse HEAD > "$@"

openfhe-install: $(OPENFHE_DST)/.build
	@ \
	$(call LD_ADD,$(OPENFHE_DST)/lib); \
	export LD_LIBRARY_PATH=$${LD_LIBRARY_PATH:?}

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
	git submodule update --init --recursive "$(OPENFHE_PYTHON_SRC)"

openfhe-python-build: $(OPENFHE_PYTHON_DST)/.build
$(OPENFHE_PYTHON_DST)/.build: $(OPENFHE_PYTHON_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(OPENFHE_PYTHON_DST)" "$(OPENFHE_PYTHON_SRC)/build"
	cd "$(OPENFHE_PYTHON_SRC)/build" && \
		cmake \
			-D CMAKE_PREFIX_PATH="$(OPENFHE_DST)" \
			-D CMAKE_INSTALL_PREFIX="$(OPENFHE_PYTHON_SRC)/openfhe" \
			.. && \
		cmake --build . --parallel "$(NPROC)" && \
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
	git -C "$(OPENFHE_PYTHON_SRC)" rev-parse HEAD > "$@"

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
	git submodule update --init --recursive "$(UARCHFHE_SRC)"

uarchfhe-build: $(UARCHFHE_DST)/.build
$(UARCHFHE_DST)/.build: $(UARCHFHE_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(UARCHFHE_DST)"
	cd "$(UARCHFHE_SRC)/crates/fhe_py_binding" && \
		python3 -m build -wo "$(UARCHFHE_DST)"
	git -C "$(UARCHFHE_SRC)" rev-parse HEAD > "$@"

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
	pydtnn-test \
	pydtnn-lint \
	pydtnn-clean

pydtnn: pydtnn-build

pydtnn-deps:
	$(APT) install -y python3 gcc patchelf
	$(PIP) install build auditwheel

pydtnn-src: $(PYDTNN_SRC)/.git
$(PYDTNN_SRC)/.git:
	git clone "https://github.com/hpca-uji/PyDTNN.git" .

pydtnn-build: $(PYDTNN_DST)/.build
$(PYDTNN_DST)/.build: $(PYDTNN_SRC)/.git | $(DST)/.gitignore
	mkdir -p "$(PYDTNN_DST)"
	TMPDIR=$$(mktemp -d) && \
	trap "rm -r $${TMPDIR:?}" EXIT && \
	cd "$(PYDTNN_SRC)" && \
		python3 -m build -so "$(PYDTNN_DST)" && \
		python3 -m build -wo "$${TMPDIR:?}" && \
		python3 -m auditwheel repair -w "$(PYDTNN_DST)" "$${TMPDIR:?}"/pydtnn-*.whl
	git -C "$(PYDTNN_SRC)" rev-parse HEAD > "$@"

pydtnn-install: PYDTNN_PKG :=
pydtnn-install: $(PYDTNN_DST)/.build
	WHEEL=$$(printf '%s ' "$(PYDTNN_DST)"/pydtnn-*.whl) && \
	$(PIP) install "$${WHEEL:?}$(if $(PYDTNN_PKG),[$(PYDTNN_PKG)])"

pydtnn-develop: PYDTNN_PKG := dev
pydtnn-develop:
	$(PIP) install \
		--config-settings editable_mode=compat \
		-e "$(PYDTNN_SRC)$(if $(PYDTNN_PKG),[$(PYDTNN_PKG)])"

pydtnn-format: PYDTNN_PKG := pydtnn
pydtnn-format:
	"$(PYDTNN_SRC)/scripts/srcs/format.sh" "$(PYDTNN_PKG)"

pydtnn-test: PYDTNN_PKG := pydtnn.tests.groups.all
pydtnn-test:
	mkdir -p "$(PYDTNN_DST)"
	TMPDIR=$$(mktemp -d) && \
	trap "rm -r $${TMPDIR:?}" EXIT && \
	cd "$${TMPDIR:?}" && \
		pytest -v -n "$(NPROC)" \
			--junitxml="$(PYDTNN_DST)/tests.xml" \
			--cov=pydtnn --cov-config="$(PYDTNN_SRC)/pyproject.toml" \
			--cov-report=term --cov-report=xml:"$(PYDTNN_DST)/coverage.xml" \
			--pyargs "$(PYDTNN_PKG)"

pydtnn-lint: PYDTNN_PKG := pydtnn
pydtnn-lint:
	mkdir -p "$(PYDTNN_DST)"
	cd "$(PYDTNN_SRC)" && \
		flake8 \
			--tee --exit-zero \
			--jobs "$(NPROC)" \
			--format=gl-codeclimate \
			--output-file="$(PYDTNN_DST)/quality.json" \
			"$(PYDTNN_PKG)"

pydtnn-clean:
	cd "$(PYDTNN_SRC)" && \
		rm -rf "$(PYDTNN_DST)" && \
		find "$(PYDTNN_SRC)/pydtnn" \
			-iname "*.pyc" \
			-iname "*.so" \
			-iname "*.dll" \
			-iname "*.dsym" \
			-delete
