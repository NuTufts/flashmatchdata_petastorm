#!/bin/bash

export GEOMLOSS_DIR=${PWD}/dependencies/geomloss
export SIREN_DIR=${PWD}/dependencies/siren-pytorch

# add model folder to python path
[[ ":$PYTHONPATH:" != *":${GEOMLOSS_DIR}:"* ]] && export PYTHONPATH="${GEOMLOSS_DIR}:${PYTHONPATH}"
[[ ":$PYTHONPATH:" != *":${SIREN_DIR}:"* ]] && export PYTHONPATH="${SIREN_DIR}:${PYTHONPATH}"
