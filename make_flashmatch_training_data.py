from __future__ import print_function
import os,sys,argparse
from math import fabs

parser = argparse.ArgumentParser(description='Make MC flashmatch training data from ROOT file. Store into petastorm.')
parser.add_argument('-db',"--db-folder",required=True,type=str,help="path to directory storing PySpark database")
parser.add_argument('-lcv',"--in-larcvtruth",required=True,type=str,help="path to larcv truth root file")
parser.add_argument('-mc',"--in-mcinfo",required=True,type=str,help="path to mcinfo root file")
parser.add_argument('-op',"--in-opreco",required=True,type=str,help="path to opreco root file")
parser.add_argument('-v',"--verbosity",type=int,default=0,help='Set Verbosity Level [0=quiet, 2=debug]')
parser.add_argument('-e',"--entry",type=int,default=None,help='Run specific entry')
args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
from ROOT import std

rt.gStyle.SetOptStat(0)
rt.gROOT.ProcessLine( "gErrorIgnoreLevel = 3002;" )

import numpy as np
from pyspark.sql import SparkSession
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

import flashmatchnet
from flashmatchnet.data.petastormschema import FlashMatchSchema
from sa_table import get_satable_maxindices

"""
test script that demos the Flash Matcher class.
"""

### DEV OUTPUTS
output_url="file:///"+args.db_folder
WRITE_TO_SPARK = True # For debug. Turn off to avoid modifying database

start_entry = 0
end_entry = -1
if args.entry is not None:
    start_entry = args.entry
    end_entry = start_entry+1

sourcefile = os.path.basename(args.in_mcinfo)
input_larcv_rootfile_v = [args.in_larcvtruth]
input_larlite_rootfile_v = [args.in_opreco,args.in_mcinfo]

adc_name = "wiremc"

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
fmutil = larflow.opticalmodel.OpModelMCDataPrep()
fmutil.setVerboseLevel(args.verbosity)

# drift velocity
from ROOT import larutil
larp = larutil.LArProperties.GetME()
driftv = larp.DriftVelocity()

# c++ classes that provides spacepoint labels
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxel_len = 5.0
voxelizer.set_voxel_size_cm( voxel_len ) # re-define voxels to 5 cm spaces
ndims_v = voxelizer.get_dim_len()
origin_v = voxelizer.get_origin()
tpc_origin = std.vector("float")(3)
tpc_origin[0] = 0.0
tpc_origin[1] = -117.0
tpc_origin[2] = 0.0

tpc_end = std.vector("float")(3)
tpc_end[0] = 256.0
tpc_end[1] = 117.0
tpc_end[2] = 1036.0

index_tpc_origin = [ voxelizer.get_axis_voxel(i,tpc_origin[i]) for i in range(3) ]
index_tpc_end    = [ voxelizer.get_axis_voxel(i,tpc_end[i]) for i in range(3) ]

print("VOXELIZER SETUP =====================")
print("origin: (",origin_v[0],",",origin_v[1],",",origin_v[2],")")
print("ndims: (",ndims_v[0],",",ndims_v[1],",",ndims_v[2],")")
print("index-tpc-origin: ",index_tpc_origin)
print("index-tpc-end: ",index_tpc_end)

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward )
for f in input_larcv_rootfile_v:
    iolcv.add_in_file( f )
iolcv.specify_data_read( larcv.kProductImage2D,  "wire" )
iolcv.specify_data_read( larcv.kProductImage2D,  "wiremc" )
iolcv.specify_data_read( larcv.kProductChStatus, "wire" )
iolcv.specify_data_read( larcv.kProductChStatus, "wiremc" )
iolcv.specify_data_read( larcv.kProductImage2D,  "ancestor" )
iolcv.specify_data_read( larcv.kProductImage2D,  "instance" )
iolcv.specify_data_read( larcv.kProductImage2D,  "segment" )
iolcv.specify_data_read( larcv.kProductImage2D,  "larflow" )
iolcv.reverse_all_products()
iolcv.initialize()

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
for f in input_larlite_rootfile_v:
    ioll.add_in_filename( f )    
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
ioll.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
ioll.open()

nentries = iolcv.get_n_entries()
print("Number of entries (LARCV): ",nentries)
print("Number of entries (LARLITE): ",ioll.get_entries())

if WRITE_TO_SPARK:
    spark_session = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark_session.sparkContext

    # remove past chunk from database
    chunk_folder = args.db_folder+"/\'sourcefile=%s\'"%(sourcefile)
    if os.path.exists(chunk_folder):
        print("remove old database folder")
        os.system("rm -r %s"%(chunk_folder))

if end_entry<0:
    end_entry = nentries

row_data = []
    
for ientry in range( start_entry, end_entry ):

    print()
    print("==========================")
    print("===[ EVENT ",ientry," ]===")

    fmutil.clear();

    iolcv.read_entry(ientry)
    ioll.go_to(ientry)
    
    run     = ioll.run_id()
    subrun  = ioll.subrun_id()
    eventid = ioll.event_id()

    truth_correct_tdrift = True
    voxelizer.process_fullchain_withtruth( iolcv, ioll, adc_name, adc_name, truth_correct_tdrift )
    
    # match reco flashes to true track and shower information
    print("Run opdataprep")    
    fmutil.process( ioll, voxelizer )
    fmutil.printMatches()
    #opdataprep.printFiltered()
    #opdataprep.tagBadFlashMatches( voxelizer, ioll )

    fulldata = voxelizer.make_voxeldata_dict()
    print("full data ----------------")
    print("  voxel coord: ",fulldata['voxcoord'].shape)
    print("  coord x-bounds: [",fulldata['voxcoord'][:,0].min(),",",fulldata['voxcoord'][:,0].max(),"]")
    print("  coord y-bounds: [",fulldata['voxcoord'][:,1].min(),",",fulldata['voxcoord'][:,1].max(),"]")
    print("  coord z-bounds: [",fulldata['voxcoord'][:,2].min(),",",fulldata['voxcoord'][:,2].max(),"]")
    print("--------------------------")

    naccepted = 0

    for iflash in range( fmutil.recoflash_v.size() ):

        flash = fmutil.recoflash_v.at(iflash)
    
        # get flash match vectors
        #coord_v = std.vector("std::vector<int>")()
        #feat_v  = std.vector("std::vector<float>")()
        #opdataprep.getChargeVoxelsForFlash( flash, voxelizer, coord_v, feat_v )

        # get the right flash pe vector
        #if flash.producerid>=0:
        #    flash_np = flash_np_v[ (flash.producerid,flash.index) ]
        #else:
        #    flash_np = np.zeros( 32, dtype=np.float32 )            
        
        data_dict = fmutil.make_opmodel_data_dict( flash, voxelizer, ioll )
        
        print("flash[",iflash,"]")
        print("  ",fmutil.strRecoMatchInfo( flash, iflash ))
        print("  pe: ",data_dict["flashpe"].shape)        
        print("  voxel coord: ",data_dict["voxcoord"].shape)
        print("  voxel charge: ",data_dict["voxcharge"].shape)
        print("  coord x-bounds: [",data_dict['voxcoord'][:,0].min(),",",data_dict['voxcoord'][:,0].max(),"]")
        print("  coord y-bounds: [",data_dict['voxcoord'][:,1].min(),",",data_dict['voxcoord'][:,1].max(),"]")
        print("  coord z-bounds: [",data_dict['voxcoord'][:,2].min(),",",data_dict['voxcoord'][:,2].max(),"]")        
        print("  frac of track traj. in tpc with voxel charge: ",fmutil.flash_track_frac_intpc_w_charge.at(iflash))            

        # we have to use the true t0 time and remove the shift
        # shift should occur when making voxels
        t0shift_cm = (flash.tick-3200.0)*0.5*driftv
        t0shift_vox = int(t0shift_cm/voxel_len)

        # we are missing the drift time
        # thus, can only know true position in data for training the model
        # if when we subtract the t0shift
        # the end of the tracks are at the anode or cathode
        vox_xmin = data_dict['voxcoord'][:,0].min()
        vox_xmax = data_dict['voxcoord'][:,0].max()

        # remove t0shift
        print("  t0shift_cm: ",t0shift_cm)
        print("  t0shift_vox: ",t0shift_vox)

        # is xmin-t0shift close to zero?
        # is xmax-t0shift close to 256?
        anode_dt = vox_xmin-t0shift_vox
        cathode_dt = (vox_xmax-(t0shift_vox+256.0/voxel_len))
        print("  anode_dt: ",anode_dt)
        print("  cathode_dt: ",cathode_dt)

        keep = False
        if abs(anode_dt)<=2:
            win_xmin = vox_xmin
            win_xmax = win_xmin + int(260.0/voxel_len)
            keep = True
            print("  detected as anode-crossing")
        elif abs(cathode_dt)<=4:
            win_xmax = vox_xmax
            win_xmin = vox_xmax - int(260.0/voxel_len)
            keep = True
            print("  detected as cathode-crossing")
        elif flash.producerid==0:
            print("cheating: saving neutrinos")
            keep = True
            win_xmin = index_tpc_origin[0]
            win_xmax = index_tpc_end[0]

        if not keep:
            print("  neither anode or cathode crossing")
            continue

        print("  shifted tpc bounds in x: [",win_xmin,",",win_xmax,"]")
        
        # crop around tpc
        voxcoord_above_xbound = (data_dict['voxcoord'][:,0]>=win_xmin)*(data_dict['voxcoord'][:,0]<=win_xmax)
        voxcoord_above_ybound = (data_dict['voxcoord'][:,1]>=index_tpc_origin[1])*(data_dict['voxcoord'][:,1]<=index_tpc_end[1]+1)
        voxcoord_above_zbound = (data_dict['voxcoord'][:,2]>=index_tpc_origin[2])*(data_dict['voxcoord'][:,2]<=index_tpc_end[2]+1)
        intpc = voxcoord_above_xbound*voxcoord_above_ybound*voxcoord_above_zbound
        voxcoord_intpc = data_dict['voxcoord'][intpc[:],:]
        voxfeat_intpc  = data_dict['voxcharge'][intpc[:],:]
        voxcoord_intpc[:,0] += t0shift_vox
        print("  shifted the xindex: ",voxcoord_intpc[:,0].min(),",",voxcoord_intpc[:,0].max())
        print("  voxcoord IN TPC: ",voxcoord_intpc.shape," ",voxcoord_intpc.dtype)
        print("  voxfeat IN TPC: ",voxfeat_intpc.shape," ",voxfeat_intpc.dtype)
        
        
        # make the row of data
        row = {"sourcefile":sourcefile,
               "run":run,
               "subrun":subrun,
               "event":eventid,               
               "matchindex":int(naccepted),               
               "flashpe":data_dict['flashpe'],
               "coord":voxcoord_intpc,
               "feat":voxfeat_intpc,               
               "ancestorid":int(flash.ancestorid)}
        naccepted += 1
        row_data.append( dict_to_spark_row(FlashMatchSchema,row) )

    print("number of training examples made: ",naccepted)

#end of event loop

if WRITE_TO_SPARK:
    print("===========  Write data to spark file  ================")
    print("number of rows: ",len(row_data))
    
    rowgroup_size_mb=256
    write_mode='append'
    with materialize_dataset(spark_session, output_url, FlashMatchSchema, rowgroup_size_mb):
        print("store rows to parquet file")
        spark_session.createDataFrame( row_data, FlashMatchSchema.as_spark_schema() ) \
                     .coalesce( 1 ) \
                     .write \
                     .partitionBy('sourcefile') \
                     .mode(write_mode) \
                     .parquet( output_url )
        print("spark write operation")
        
print("=== FIN ==")
