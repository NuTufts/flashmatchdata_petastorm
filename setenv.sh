
GEOMLOSS_DIR=${PWD}/dependencies/geomloss

# add model folder to python path
[[ ":$PYTHONPATH:" != *":${GEOMLOSS_DIR}:"* ]] && PYTHONPATH="${GEOMLOSS_DIR}:${PYTHONPATH}"
