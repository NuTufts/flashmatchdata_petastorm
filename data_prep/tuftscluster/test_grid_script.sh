#!/bin/bash

samplename=mcc9_v29e_dl_run3_G1_extbnb
inputlist=$PWD/filelist_mcc9_v29e_dl_run3_G1_extbnb_dlana.txt
fileidlist=$PWD/runid_mcc9_v29e_dl_run3_G1_extbnb_dlana.txt

./run_gridjob_dataprep.sh 0 1 $samplename merged_dlana $inputlist $fileidlist
