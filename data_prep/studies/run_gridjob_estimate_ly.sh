#!/bin/bash

#!/bin/bash

JOBSTARTDATE=$(date)


# Location of the repo being used
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
FLASHMATCH_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/studies/

#echo "SETUP CONTAINER/ENVIRONMENT"
cd ${UBDL_DIR}
alias python=python3
cd $UBDL_DIR
source setenv_py3_container.sh
source configure_container.sh

cd ${FLASHMATCH_DIR}
source setenv_flashmatchdata.sh
cd ${WORKDIR}

FILELIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/flashmatchdata_petastorm/data_prep/fulldataset_no_anode_mcc9_v29e_dl_run3_G1_extbnb.txt
OUTFILE=output_lyana_no_anode_mcc9_v29e_dl_run3_G1_extbnb.root

python3 estimate_initial_ly.py -i ${FILELIST} -o ${OUTFILE}

