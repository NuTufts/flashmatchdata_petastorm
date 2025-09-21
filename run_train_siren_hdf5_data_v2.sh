#!/bin/bash

UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl
WORK_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/

cd $UBDL_DIR

source setenv_no_libtorch.sh
source configure_container.sh

cd $WORK_DIR
source setenv_flashmatchdata.sh

# config_siren_hdf5_data.yaml
python3 train_siren_hdf5_data_v2.py > log_train_siren_hdf5_data_v2_no_anode.txt 
