import os,sys,argparse

import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)
rt.gROOT.ProcessLine( "gErrorIgnoreLevel = 3002;" )

import numpy as np
from pyspark.sql import SparkSession

from flashmatchdata import process_one_entry, write_event_data_to_spark_session


"""
test script that demos the Flash Matcher class.
"""

### DEV OUTPUTS
output_url="file:///tmp/test_flash_dataset"
nentries = 5
WRITE_TO_SPARK = True

### DEV INPUTS
##dlmerged = "dlmerged_mcc9_v13_bnbnue_corsika.root"
##dlmerged = "testfile_01.root"
dlmerged = "testfile_02.root"
start_entry = 0
end_entry = -1

input_rootfile_v = [dlmerged]

# what we will extract:
# we need to flash match mctracks to optical reco flashes
# interestingly we also have the waveforms here I think
# the triplet truth-matching code should cluster triplets by ancestor ID.
# then we need to voxelize
# then we can store for each entry
#   1) coordinate tensor (N,3)
#   2) charge vector for each filled voxel (N,3)
#   3) flashmatched optical reco vector


# flashmatch class from ublarcvapp
fmutil = ublarcvapp.mctools.FlashMatcherV2()
fmutil.setVerboseLevel(0)

# c++ classes that provides spacepoint labels
#tripmaker = larflow.prep.PrepMatchTriplets()
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelizer.set_voxel_size_cm( 5.0 ) # re-define voxels to 5 cm spaces

io = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )
for f in input_rootfile_v:
    io.add_in_file( f )
io.specify_data_read( larcv.kProductImage2D,  "wire" )
io.specify_data_read( larcv.kProductImage2D,  "wiremc" )
io.specify_data_read( larcv.kProductChStatus, "wire" )
io.specify_data_read( larcv.kProductChStatus, "wiremc" )
io.specify_data_read( larcv.kProductImage2D,  "ancestor" )
io.specify_data_read( larcv.kProductImage2D,  "instance" )
io.specify_data_read( larcv.kProductImage2D,  "segment" )
io.specify_data_read( larcv.kProductImage2D,  "larflow" )
io.reverse_all_products()
io.initialize()

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
for f in input_rootfile_v:
    ioll.add_in_filename( f )    
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
ioll.open()

nentries = ioll.get_entries()
print("Number of entries: ",nentries)

if WRITE_TO_SPARK:
    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark.sparkContext

if end_entry<0:
    end_entry = nentries

row_data = []
    
for ientry in range( start_entry, end_entry ):

    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")

    io.read_entry(ientry)
    ioll.go_to(ientry)
    #opio.go_to(ientry)
    
    run     = ioll.run_id()
    subrun  = ioll.subrun_id()
    eventid = ioll.event_id()

    event_row_data = process_one_entry( os.path.basename(input_rootfile_v[0]), ientry, io, ioll, fmutil, voxelizer )
    row_data += event_row_data

    #print(row_data)

write_event_data_to_spark_session( spark, output_url, row_data )



print("=== FIN ==")
