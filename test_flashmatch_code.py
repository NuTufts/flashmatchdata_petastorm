import os,sys,argparse

import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)
rt.gROOT.ProcessLine( "gErrorIgnoreLevel = 3002;" )

import numpy as np

from flashmatchdata import process_one_entry


"""
test script that demos the Flash Matcher class.
"""

### DEV INPUTS
dlmerged = "dlmerged_mcc9_v13_bnbnue_corsika.root"

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


#opio = larlite.storage_manager( larlite.storage_manager.kREAD )
#opio.add_in_filename(  args.input_opreco )
#opio.open()

#f = TFile(args.input_voxelfile,"READ")
#print("passed tfile part")

#voxio = larlite.storage_manager( larlite.storage_manager.kREAD )
#voxio.add_in_filename(  args.input_voxelfile )
#voxio.open()

#outio = larlite.storage_manager( larlite.storage_manager.kWRITE )
#outio.set_out_filename(  args.output_file )
#outio.open()

nentries = ioll.get_entries()
print("Number of entries: ",nentries)

#print("Start loop.")

#fmutil.initialize( voxio )


start_entry = 0
end_entry = 1
if "_v2" in dlmerged:
    start_entry = 10
    end_entry = 15

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

    row_data = process_one_entry( os.path.basename(input_rootfile_v[0]), ientry, io, ioll, fmutil, voxelizer )
    print(row_data)

    sys.exit(-1)

    # Get the first entry (or row) in the tree (i.e. table)
    kploader.load_entry(ientry)

    # turn shuffle off (to do, function should be kploader function)
    tripdata = kploader.triplet_v.at(0).setShuffleWhenSampling( False )

    # 2d images
    wireimg_dict = {}
    for p in range(3):
        wireimg = kploader.triplet_v.at(0).make_sparse_image( p )
        wireimg_coord = wireimg[:,:2].astype(np.long)
        wireimg_feat  = wireimg[:,2]
        wireimg_dict["wireimg_coord%d"%(p)] = wireimg_coord
        wireimg_dict["wireimg_feat%d"%(p)] = wireimg_feat        

    # get 3d spacepoints (to do, function should be kploader function)
    tripdata = kploader.triplet_v.at(0).get_all_triplet_data( True )
    spacepoints = kploader.triplet_v.at(0).make_spacepoint_charge_array()    
    nfilled = c_int(0)
    ntriplets = tripdata.shape[0]    


    # reco flash vectors
    producer_v = ["simpleFlashBeam","simpleFlashCosmic"]
    flash_np_v = {}
    for iproducer,producer in enumerate(producer_v):
        
        flash_beam_v = ioll.get_data( larlite.data.kOpFlash, producer )
    
        for iflash in range( flash_beam_v.size() ):
            flash = flash_beam_v.at(iflash)

            # we need to make the flash vector, the target output
            flash_np = np.zeros( flash.nOpDets() )

            for iopdet in range( flash.nOpDets() ):
                flash_np[iopdet] = flash.PE(iopdet)

            # uboone has 4 pmt groupings
            score_group = {}
            for igroup in range(4):
                score_group[igroup] = flash_np[ 100*igroup: 100*igroup+32 ].sum()
            print(" [",producer,"] iflash[",iflash,"]: ",score_group)
            
            if producer=="simpleFlashBeam":
                flash_np_v[(iproducer,iflash)] = flash_np[0:32]
            elif producer=="simpleFlashCosmic":
                flash_np_v[(iproducer,iflash)] = flash_np[200:232]
                
    with materialize_dataset(spark, output_url, FlashMatchSchema, rowgroup_size_mb):
        
        print("assemble row data")
        iindex = 0
        rows_dd = []
        for k,v in flash_np_v.items():
            print("store flash: ",k," ",v.sum())
            row = {"sourcefile":dlmerged,
                   "run":int(ioll.run_id()),
                   "subrun":int(ioll.subrun_id()),
                   "event":int(ioll.event_id()),
                   "trackindex":int(iindex),
                   "array_flashpe":v}
            rows_dd.append( dict_to_spark_row(FlashMatchSchema,row) )
            iindex += 1
        print("store rows to parquet file")
        spark.createDataFrame(rows_dd, FlashMatchSchema.as_spark_schema() ) \
            .coalesce( 1 ) \
            .write \
            .partitionBy('sourcefile') \
            .mode('append') \
            .parquet( output_url )
        print("spark write operation")


print("=== FIN ==")
