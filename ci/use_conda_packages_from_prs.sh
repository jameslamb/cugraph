#!/bin/bash

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)

CUDF_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 cpp)
CUDF_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 python)

UCXX_CHANNEL=$(rapids-get-pr-conda-artifact ucxx 278 cpp)

LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 python)

# NOTE: rapids-get-pr-wheel-artifact seems to not work with private repos
# LIBCUGRAPHOPS_CHANNEL=$(rapids-get-pr-conda-artifact cugraph-ops 692 cpp)
# PYLIBCUGRAPHOPS_CHANNEL=$(rapids-get-pr-conda-artifact cugraph-ops 692 python)

conda config --system --add channels "${LIBRMM_CHANNEL}"
conda config --system --add channels "${RMM_CHANNEL}"
conda config --system --add channels "${CUDF_CPP_CHANNEL}"
conda config --system --add channels "${CUDF_PYTHON_CHANNEL}"
conda config --system --add channels "${UCXX_CHANNEL}"
conda config --system --add channels "${LIBRAFT_CHANNEL}"
conda config --system --add channels "${RAFT_CHANNEL}"
# conda config --system --add channels "${LIBCUGRAPHOPS_CHANNEL}"
# conda config --system --add channels "${PYLIBCUGRAPHOPS_CHANNEL}"
