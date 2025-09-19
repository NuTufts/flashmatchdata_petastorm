Preparing corsika to have ssnet and process through lantern workflow

We use the lantern container.

Local location:

```
~/working/larbys/larbys-containers/lantern_cpuonly/lantern_v2_me_06_03_prod/
```

To start:
```
singularity shell ~/working/larbys/larbys-containers/lantern_cpuonly/lantern_v2_me_06_03_prod/
```

Setup environment:
```
source /cluster/home/lantern_scripts/setup_lantern_container.sh
```

First, we run sparse uresnet. we also pass through larcv truth images so that the output is Tick Forwards
```
python3 inference_sparse_ssnet_uboone.py -i merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root -w $SSNET_DIR/weights -o ssnet_output.root -tb
```

Then, we make 2d shower (index0) and track (index1) images for each plane
```
python3 ${SSNET_DIR}/recreate_ubspurn.py -i ssnet_output.root -o ssnet_ubspurn_output.root
```

Next, hadd to make larcv file
```
hadd merged_dlreco_with_ssnet.root ssnet_output.root ssnet_ubspurn_output.root
```

Add larlite info to make merged_dlreco file:
```
hadd -f merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root merged_dlreco_with_ssnet.root old/opreco-Run000001-SubRun000001.root old/reco2d-Run000001-SubRun000001.root
```

We need to remove the larlite_id_tree in this merged file, because it will have duplicated the id tree from opreco and reco2d
```
rootrm merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root:larlite_id_tree
```

We copy the larlite_id_tree from opreco
```
rootcp old/opreco-Run000001-SubRun000001.root:larlite_id_tree merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root
```

Now we can run larmatch

```
python3 ${LARMATCH_DIR}/deploy_larmatchme.py --config-file /cluster/home/lantern_scripts/config_larmatchme_deploycpu.yaml --supera merged_dlreco_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root --weights /cluster/home/ubdl/larflow/larmatchnet/larmatch/larmatch_ckpt78k.pt --output larmatch_test_corsika.root --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu --use-skip-limit
```

Before we can run the reco, we need to make a fake thrumu image, which we will do by cheating and using the ancestor image to tag neutrino-origin pixels.
I NEED TO WRITE THIS
```
```

Finally, we can run reco
```
```

We can also run the cosmic reco
```
```

We can make MC based flash-match data with the cosmic reco.