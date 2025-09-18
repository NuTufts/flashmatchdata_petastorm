#!/bin/bash

GEOMLOSS_DIR=${PWD}/dependencies/geomloss
SIREN_DIR=${PWD}/dependencies/siren-pytorch

export FLASHMATCH_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export FLASHMATCH_LIBDIR=${FLASHMATCH_BASEDIR}/data_prep/build/installed/lib
export FLASHMATCH_INCDIR=${FLASHMATCH_BASEDIR}/data_prep/build/installed/include
export FLASHMATCH_BINDIR=${FLASHMATCH_BASEDIR}/data_prep/build/installed/bin
[[ ":$LD_LIBRARY_PATH:" != *":${FLASHMATCH_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${FLASHMATCH_LIBDIR}:${LD_LIBRARY_PATH}"
[[ ":$PATH:" != *":${FLASHMATCH_BINDIR}:"* ]] && PATH="${FLASHMATCH_BINDIR}:${PATH}"

# add model folder to python path
[[ ":$PYTHONPATH:" != *":${GEOMLOSS_DIR}:"* ]] && PYTHONPATH="${GEOMLOSS_DIR}:${PYTHONPATH}"
[[ ":$PYTHONPATH:" != *":${SIREN_DIR}:"* ]] && PYTHONPATH="${SIREN_DIR}:${PYTHONPATH}"

