from __future__ import print_function
import os,sys,argparse,signal
from math import fabs

parser = argparse.ArgumentParser(description='Make MC flashmatch training data from ROOT file. Store into petastorm.')
parser.add_argument('-db',"--db-folder",required=True,type=str,help="path to directory storing PySpark database")
parser.add_argument('-lcv',"--in-larcvtruth",required=True,type=str,help="path to larcv truth root file")
parser.add_argument('-mc',"--in-mcinfo",required=True,type=str,help="path to mcinfo root file")
parser.add_argument('-op',"--in-opreco",required=True,type=str,help="path to opreco root file")
parser.add_argument('-v',"--verbosity",type=int,default=0,help='Set Verbosity Level [0=quiet, 2=debug]')
parser.add_argument('-e',"--entry",type=int,default=None,help='Run specific entry')
parser.add_argument('-n',"--num-entries",type=int,default=None,help='Run n entries')
parser.add_argument('-xw',"--no-write",default=False,action='store_true',help="If flag given, we will not write to DB. For debugging.")
parser.add_argument('-ow',"--over-write",default=False,action='store_true',help="If flag given, will overwrite existing database chunk without user check")
parser.add_argument('-p',"--port",type=int,default=4000,help="Set starting port number for spark web UI")
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
WRITE_TO_SPARK = not args.no_write # For debug. Turn off to avoid modifying database
ONLY_ANODE_CATHODE = False
OVERWRITE_INPUT_TIMEOUT_SECS = 10

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

start_entry = 0
end_entry = nentries
if args.entry is not None:
    start_entry = args.entry
    end_entry = start_entry+1
if args.num_entries is not None:
    if end_entry<0:
        end_entry = nentries
    else:
        end_entry = start_entry + args.num_entries
        if end_entry>nentries:
            end_entry = nentries

print("RUN FROM ENTRY ",start_entry," to ",end_entry)


if WRITE_TO_SPARK:
    print("********** WRITING TO SPARK DB ***************")
    spark_session = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').config("spark.ui.port", "%d"%(args.port)).getOrCreate()
    sc = spark_session.sparkContext

    # remove past chunk from database
    chunk_folder = args.db_folder+"/\'sourcefile=%s\'"%(sourcefile)
    if os.path.exists(chunk_folder):
        print("*****  removing old database chunk folder: ",chunk_folder,"  **********")
        if args.over_write:
            print("proceeding without check due to '--over-write' flag has been given")
            os.system("rm -r %s"%(chunk_folder))
        else:
            print("[enter 'y' or 'Y' to allow overwrite. any other input stops program.] (to over-write without check, give '--over-write')")
            # defining a handler
            def handle_no_input(signum,frame):
                raise IOError("user input check timed out")
            signal.signal(signal.SIGALRM, handle_no_input)
            
            signal.alarm(OVERWRITE_INPUT_TIMEOUT_SECS)
            userinput = 'n'
            try:
                userinput = input()
            except:
                print("over-write check timed out. stopping program.")
                sys.exit(1)
            signal.alarm(0) # disable alarm after success
            # should be a check here unless overwrite argument given
            if userinput in ['y','Y']:
                print("user OK( entered: %s ) provided to delete chunk folder"%(userinput))
                os.system("rm -r %s"%(chunk_folder))
            else:
                print("overwrite not allowed by user input: ",userinput)
                sys.exit(0)
        


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

    # the function below uses the wire plane images in stored in the larcv io manager (iolcv)
    # and proposes possible 3d spacepoints consistent with the pattern of ionization in the images
    # it then uses the 'truth' information in the simulation to label those proposed spacepoints
    # Labels relevant here are:
    #  1. is the spacepoint real or a 'ghost' point?
    #  2. the ID number from Geant4 associated to the particle that
    #       left the ionization deposite the spacepoint represents
    #  3. the ID number of the "ancestor" particle that ultimately led to the current particle
    #  4. The value of the pixels on each plane associated to each point.
    # After labels for all the points are made, we define voxels over a 3D rectangle and assign
    #     points to individual voxels in the grid.
    #     (A voxel may be assigned more than one point.)
    #     (each point is assigned to only one voxel)
    # Transferring labels to the voxels requires some may to reduce the set of labels from the points.
    # For a given voxel, with a set of points
    #  1. if any point has a 'true' label, the voxel is labeled as 'true'
    #  2. for our purposes, we will not need to reduce track id labels. we will need a set of voxels    
    #     to store the ionization deposited by one 'interaction cascade'. A cascade is paired
    #     to a reconstructed optical flash in order to train our network. When we select the set of
    #     voxels to represent the ionization, we can take all voxels associated to points carrying
    #     the ancestor ID of the interaction cascade.
    # One complication is that particles coming from a neutrino interaction have different
    #  ancestor IDs because they are given as 'primary' particles to Geant4. We change
    #  ancestor IDs for particles coming from neutrino interactions to a common ID of '0'.
    #
    # Note: if 'truth_correct_tdrift' is TRUE, then we will use the truth information to
    #   remove the t0-offset for each point.
    # We can do this, beacause a true point (usually) is labeled by a track ID.
    # We can then look up the true time the particle crossed the detector and subtract the 
    #   the x-position offset by (t_particle - t_trigger)*drift_velocity.
    #   (We use units of usec and cms).
    # We make this position adjustment to points BEFORE assigning them to voxels.
    # The result is that all voxels should be within the voxels corresponding to the drift volume
    #   in time with the beam.
    truth_correct_tdrift = True    
    voxelizer.process_fullchain_withtruth( iolcv, ioll, adc_name, adc_name, truth_correct_tdrift )

    # for debug: dump out the particle tree graph, showing the particles whose information we have.
    #mcpg = ublarcvapp.mctools.MCPixelPGraph()
    #mcpg.buildgraphonly( ioll )
    #mcpg.printGraph(0,False)
    
    # match reco flashes to true track and shower information.
    # this version uses the voxel information as well to
    # check if examples are not missing a significant fraction of
    # voxels with charge.
    # Note: the process function needs to know if the voxels already have their
    # true t0 shift removed!!!
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

    # loop through the truth-matched reconstruction optical flashes
    # that have been matched to a set of particles in the saved simulation info.
    for iflash in range( fmutil.recoflash_v.size() ):

        flash = fmutil.recoflash_v.at(iflash)
    
        # the following function provides us with the info we need.
        # below returns a dictionary with keys:
        # 'flashpe':   (1,32) float32 array with total PE in light pulse seen by each PMT
        # 'voxcoord':  (N,3) int64 array with the index of the N voxels that contain space points
        # 'voxcharge': (N,3) float32 array with the charge sum of points for each plane.
        #              For each plane, an individual pixel is only able to contribute once to the total.
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

        # get the t_drift adjustment.
        # emperical way is to use anode or cathode crossers only
        # or we can use the truth

        # skip null flash outliers for now
        if flash.producerid==-1:
            continue
        
        # since we already removed the t0shift, the voxels should be in the right place
        # just crop in the main drift window
        win_xmin = index_tpc_origin[0] 
        win_xmax = index_tpc_end[0]
        print("  shifted tpc bounds in x: [",win_xmin,",",win_xmax,"]")
        
        # crop around tpc
        # below gives me a T value for each voxel, that falls inside an interval for a given dimension.
        voxcoord_above_xbound = (data_dict['voxcoord'][:,0]>=win_xmin)*(data_dict['voxcoord'][:,0]<=win_xmax)
        voxcoord_above_ybound = (data_dict['voxcoord'][:,1]>=index_tpc_origin[1])*(data_dict['voxcoord'][:,1]<=index_tpc_end[1]+1)
        voxcoord_above_zbound = (data_dict['voxcoord'][:,2]>=index_tpc_origin[2])*(data_dict['voxcoord'][:,2]<=index_tpc_end[2]+1)
        # create a box interval by requiring that voxels fall within the intervals of all the dimensions
        intpc = voxcoord_above_xbound*voxcoord_above_ybound*voxcoord_above_zbound
        # select voxels falling within my 3D box interval representing the TPC
        voxcoord_intpc = data_dict['voxcoord'][intpc[:],:]
        voxfeat_intpc  = data_dict['voxcharge'][intpc[:],:]
        # subtract off the index value representing the lower-x corner of the TPC
        # this effectively shrinks the voxel set to only those inside the TPC
        voxcoord_intpc[:,0] -= win_xmin
        print("  x-index bounds after mask and shifted: ",voxcoord_intpc[:,0].min(),",",voxcoord_intpc[:,0].max())
        print("  voxcoord IN TPC shape and type: ",voxcoord_intpc.shape," ",voxcoord_intpc.dtype)
        print("  voxfeat IN TPC shape and type: ",voxfeat_intpc.shape," ",voxfeat_intpc.dtype)

        #print("voxcoord_intpc ===================")
        #print(voxcoord_intpc)
        #print("voxcoord_intpc ===================")        

        # if no more voxels remaining, just skip this example.
        if ( voxcoord_intpc.shape[0]<1 ):
            continue
        
        # make the row of data to save in our data
        row = {"sourcefile":sourcefile,
               "run":run,
               "subrun":subrun,
               "event":ientry,               
               "matchindex":int(naccepted),               
               "flashpe":data_dict['flashpe'],
               "coord":voxcoord_intpc,
               "feat":voxfeat_intpc,               
               "ancestorid":int(flash.ancestorid)}
        naccepted += 1
        row_data.append( dict_to_spark_row(FlashMatchSchema,row) )
        
    # end of loop over flash-voxel pairs
    print("number of training examples made in this event: ",naccepted)

#end of event loop

if WRITE_TO_SPARK:
    print("===========  Write data to spark file  ================")
    print("number of rows: ",len(row_data))
    
    rowgroup_size_mb=256
    #write_mode='overwrite'
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
else:
    print("Skipping writing to data base")
        
print("=== FIN ==")
