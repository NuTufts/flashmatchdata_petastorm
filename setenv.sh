
GEOMLOSS_DIR=${PWD}/dependencies/geomloss
SIREN_DIR=${PWD}/dependencies/siren-pytorch

# add model folder to python path
[[ ":$PYTHONPATH:" != *":${GEOMLOSS_DIR}:"* ]] && PYTHONPATH="${GEOMLOSS_DIR}:${PYTHONPATH}"
[[ ":$PYTHONPATH:" != *":${SIREN_DIR}:"* ]] && PYTHONPATH="${SIREN_DIR}:${PYTHONPATH}"
