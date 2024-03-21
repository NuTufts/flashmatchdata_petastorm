import os,sys
from larlite import larlite
from ublarcvapp import ublarcvapp
from larflow import larflow
import numpy as np
import torch
from ROOT import std

from pyspark.sql.types import IntegerType, StringType
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

import MinkowskiEngine as me

from sa_table import get_satable_maxindices, load_satable_fromnpz

"""
This python module contains code to create flash-match training data
for the light model project.

There are functions that help to extract the training data from 
various reconstruction and truth data stored in larlite/larcv ROOT files.
This  information is then repackaged into charge-flash pairs 
and stored as numpy arrays needed for training.

The arrays are stored using Petastorm, which helps us define the
PySpark database structure so that we can write and read with 
relative ease. Petastorm also includes ways to easily build
a pytorch DataLoader interface for reading the data during training.

"""


# This defines the table we will write to the database
# It's basically defining the columns of the database table,
#  listing the for each: the name, base type, shape of array, and how to pack/unpack it into bits
#  note the last pool in the tuble defining the column is whether its ok to have a missing value
#  for the column.
FlashMatchSchema = Unischema("FlashMatchSchema",[
    UnischemaField('sourcefile', np.string_, (), ScalarCodec(StringType()),  True),
    UnischemaField('run',        np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('subrun',     np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('event',      np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('matchindex', np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('ancestorid', np.int32,   (), ScalarCodec(IntegerType()), False),
    UnischemaField('coord',      np.int64,   (None,3), NdarrayCodec(), False),
    UnischemaField('feat',       np.float32, (None,3), NdarrayCodec(), False),    
    UnischemaField('flashpe',    np.float32, (32,),    NdarrayCodec(), False),
])

# a global to this module as a reference
sa_maxindices = get_satable_maxindices()
sa_coord, sa_values = load_satable_fromnpz()

def make_flashmatch_data( fm, triplet ):
    """
    Makes flash and charge pairings from merging info from
      1) ublarcvapp::mctools::FlashMatcherV2
      2) larflow::TripletMaker

    Placeholder for eventual one-stop function?
    """
    pass


def get_reco_flash_vectors( ioll ):
    """
    Gets the reconstructed flashes we will match charge clusters to.
    We get the data from the larlite data structures.
    We assume that an event has already been loaded before
      this is called.
    """
    
    # reco flash vectors
    producer_v = ["simpleFlashBeam","simpleFlashCosmic"]
    flash_np_v = {}
    for iproducer,producer in enumerate(producer_v):
        
        flash_beam_v = ioll.get_data( larlite.data.kOpFlash, producer )
    
        for iflash in range( flash_beam_v.size() ):
            flash = flash_beam_v.at(iflash)

            # we need to make the flash vector, the target output
            flash_np = np.zeros( flash.nOpDets(), dtype=np.float32 )
            
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
    return flash_np_v
    

def process_one_entry( filename, ientry, iolcv, ioll, fmbuilder, voxelizer,
                       min_charge_voxels=1,
                       max_frac_out_of_tpc=0.3,
                       pezero_q_threshold=2500.0,
                       pezero_x_threshold=230.0):
    """
    process one ROOT file entry.
    In the ROOT file, one entry is one event trigger, which contains many 
      pairs of ionization clusters and flashes.
    For each event, we return with a list of rows for our
      flashmatch training data database.
    One row is one charge and flash pair, to be used for training the light model.

    We assume that the larcv and larlite IO interfaces (iolcv, ioll)
      have already been loaded to the same entry/event.
    """

    # get info that lets us modify the coordinate index tensors originally
    #  defined by the Voxelizer coordinates to the coordinates used by
    #  Polina's solid angle calculations.

    sa_maxdims = get_satable_maxindices()

    # where is the TPC origin used by polina?
    tpc_origin = std.vector("float")(3)
    tpc_origin[0] = 0.3
    tpc_origin[1] = -117.0
    tpc_origin[2] = 0.3
    
    tpc_end = std.vector("float")(3)
    tpc_end[0] = 256.0
    tpc_end[1] = 117.0
    tpc_end[2] = 1035.7

    index_tpc_origin = [ int(voxelizer.get_axis_voxel(i,tpc_origin[i])) for i in range(3) ]
    index_tpc_end    = [ int(voxelizer.get_axis_voxel(i,tpc_end[i])) for i in range(3) ]
    print("[process one entry] index_tpc_origin: ",index_tpc_origin)
    
    adc_name = "wiremc"

    run = int(ioll.run_id())
    subrun = int(ioll.subrun_id())
    event = int(ioll.event_id())

    # match reco flashes to true track and shower information
    fmbuilder.process( ioll )

    # make vectors of reco opflashes
    flash_np_v = get_reco_flash_vectors( ioll )

    # build candidate spacepoints and true labels
    truth_correct_tdrift = True
    voxelizer.process_fullchain_withtruth( iolcv, ioll, adc_name, adc_name, truth_correct_tdrift )

    voxdata = voxelizer.make_voxeldata_dict()
    coord   = voxdata["voxcoord"]
    feat    = voxdata["voxfeat"]
    truth   = voxdata["voxlabel"]
    instancedict = voxelizer.make_instance_dict_labels( voxelizer._triplet_maker )
    voxseqid = instancedict["voxinstance"]    
    trackid_to_seqid = instancedict["voxinstance2id"]

    print("coord: ",coord.shape)
    print("feat: ",feat.shape)
    print("truth: ",truth.shape)
    print("voxel sequential id tensor: ",voxseqid.shape)

    # keep the non-ghost coordinates
    mask = truth[:]==1
    coord_real   = coord[ mask[:], : ]
    feat_real    = feat[ mask[:], : ]
    trackid_real = voxseqid[ mask[:] ]
    print("unique IDs: ",np.unique(trackid_real))

    print("Prepare Match Rows")
    row_data = []    
    nrejected = 0
    
    # loop over RecoFlash_t objects.
    # get the reco flash, if exists
    # get the coordinate pixels    
    for imatch in range(fmbuilder.recoflash_v.size()):

        matchdata = fmbuilder.recoflash_v.at(imatch)
        print(" MATCH[",imatch,"]: tick=",matchdata.tick," time_us=",matchdata.time_us," aid=",matchdata.ancestorid)

        # get the flash vector
        pe_v = np.zeros( 32, dtype=np.float32 )
        if matchdata.producerid>=0:
            # if producer is not -1 (null flash)
            # set the pe values to the reco pe
            flashkey = (matchdata.producerid,matchdata.index)
            pe_v = flash_np_v[flashkey]

        # get the charge vector        
        coord_list = []
        feat_list  = []
        trackid_v = matchdata.trackid_list()
        if trackid_v.size()==0:
            print("  empty track ID list. skip.")
            continue
        
        for itrackid in range(trackid_v.size()):
            trackid = trackid_v.at(itrackid)
            
            if trackid in trackid_to_seqid:
                seqid = trackid_to_seqid[trackid]
                mask_id = trackid_real[:]==seqid
                
                coord_tid = coord_real[ mask_id[:], : ]
                feat_tid  = feat_real[ mask_id[:], : ]

                coord_list.append( coord_tid )
                feat_list.append( feat_tid )
                
        # concat the coord and feat tensors
        if len(coord_list)==0:
            print("  empty coord list? skip.")            
            continue
        
        coord_full = np.concatenate( coord_list )
        feat_full  = np.concatenate( feat_list )

        # we need to remove index offsets
        for i in range(3):
            coord_full[:,i] -= index_tpc_origin[i]

        # let's make sure we do not have any charge voxels outside the  TPC/theSA lookup table
        if coord_full.shape[0]==0:
            print("match failing because there are no charge voxels")
            continue
        
        n_f = float(coord_full.shape[0])        
        below_mask = [ coord_full[:,i]>=0 for i in range(3) ]
        above_mask = [ coord_full[:,i]<int(sa_maxdims[i]) for i in range(3) ]

        good_mask = above_mask[0]*below_mask[0]
        good_mask *= above_mask[1]*below_mask[1]
        good_mask *= above_mask[2]*below_mask[2]

        coord_good = coord_full[ good_mask[:], : ]
        feat_good  = feat_full[ good_mask[:], : ]

        frac_below = [ 1.0-float(below_mask[i].sum())/n_f for i in range(3) ]
        frac_above = [ 1.0-float(above_mask[i].sum())/n_f for i in range(3) ]
        
        print("below index fraction by coords: ",frac_below)
        print("above max-index fraction by coords: ",frac_above)

        too_many_outofbounds = False
        for i in range(3):
            if frac_below[i]>max_frac_out_of_tpc or frac_above[i]>max_frac_out_of_tpc:
                too_many_outofbounds = True
                print("match failing because too many out of bounds along axis: ",i)
                print("coord_full.shape: ",coord_full.shape)
                print("dump of bad indices (post-offset): ")
                print(coord_full[ good_mask[:]==False, :])

        if coord_good.shape[0]<min_charge_voxels:
            print("match failing because too little voxels inside tpc:  ",coord_good.shape)

        if coord_good.shape[0]<min_charge_voxels or too_many_outofbounds:
            nrejected += 1
            print("match rejected ------")
            print("  ancestorid: ",matchdata.ancestorid)
            print("  time_us: ",matchdata.time_us)
            print("  tick: ",matchdata.tick)            
            continue

        # fix zeros by using mean
        q = feat_good[:,:3]
        qzero = (q==0.0)
        nzero = np.sum( qzero, axis=1 ) # tells me the number of non-zero voxels

        # remove zero voxels, keep nzero < 3
        coord_nonzero = coord_good[nzero[:]<3,:]
        feat_nonzero  = feat_good[nzero[:]<3,:]
        print("  removed ",(nzero==3).sum()," all q-zero voxels")
        print("  coord before: ",coord_good.shape," after: ",coord_nonzero.shape)
        print("  feat before: ",feat_good.shape," after: ",feat_nonzero.shape)        

        # replace q and zero and nzero tensors
        q = feat_nonzero[:,:3]
        qzero = (q==0.0)
        nzero = np.sum(qzero,axis=1)
        # fix 1 missing value
        onezero = nzero==1
        q2mean = np.sum(q,axis=1)/2.0
        for ii in range(onezero.shape[0]):
            if onezero[ii]:
                for j in range(3):
                    if q[ii,j]==0.0:
                        q[ii,j] = q2mean[ii]        
        print("  fix feature tensor with 1 missing entry: ",onezero.sum())
        
        # fix 2 missing values
        twozero = nzero==2
        q_2zero = np.sum( q, axis=1 )
        qT = np.transpose( q[twozero[:],:3], axes=(1,0) )
        qT = q_2zero[twozero[:]]
        print("  fill in values with 2 missing values: ",twozero.sum())
        qzero = (feat_nonzero[:,:3]==0.0)
        nzero = np.sum(qzero,axis=1)
        print("  post-fix: number of entries with at least one zero: ",(nzero>0).sum())

        # filter for outlier flashes
        outlier = False        
        q = feat_nonzero[:,:3]
        pe_sum = pe_v.sum()

        if pe_sum==0.0:
            xpos = coord_nonzero*5.0 # scale by 5.0 cm
            q_mean = np.mean( q, axis=1 )
            qsum = q_mean.sum()
        
            xmean = 250.0
            if qsum>0.0:
                xmean = (xpos[:,0]*q_mean).sum()/qsum

            if qsum>pezero_q_threshold or xmean<pezero_x_threshold:
                outlier = True

            print("  check for outlier status: outlier=",outlier," qsum=",qsum," xmean=",xmean," pe=sum=",pe_sum)

        if outlier:
            nrejected += 1
            print("match Reject as outlier ---- ")
            continue
        
        # make the row
        row = {"coord":coord_nonzero,
               "feat":feat_nonzero,
               "flashpe":pe_v,
               "run":run,
               "subrun":subrun,
               "event":event,
               "matchindex":int(imatch),
               "ancestorid":int(matchdata.ancestorid),
               "sourcefile":filename}
        row_data.append( dict_to_spark_row(FlashMatchSchema,row) )

    print("[prcess-one-entry] nrejected=",nrejected," nproduced=",len(row_data))
    return row_data
    

def write_event_data_to_spark_session( spark_session, output_url, row_data,
                                       rowgroup_size_mb=256, write_mode='append' ):

    with materialize_dataset(spark_session, output_url, FlashMatchSchema, rowgroup_size_mb):
        print("store rows to parquet file")
        spark_session.createDataFrame(row_data, FlashMatchSchema.as_spark_schema() ) \
                     .coalesce( 1 ) \
                     .write \
                     .partitionBy('sourcefile') \
                     .mode(write_mode) \
                     .parquet( output_url )
        print("spark write operation")


def _default_transform_row( row ):
    #print(row)
    # original tensors from database
    coord = row['coord']
    feat  = row['feat']
    #print("[row-tranform] (pre-max index mask) coord: ",coord.shape)
    
    goodmask_i = coord[:,0]<sa_maxindices[0]
    goodmask_j = coord[:,1]<sa_maxindices[1]
    goodmask_k = coord[:,2]<sa_maxindices[2]
    goodmask = goodmask_i*goodmask_j*goodmask_k

    #print("[row-transform] num_good=",goodmask.sum())

    coord = coord[goodmask[:],:] # remove bad rows
    feat  = feat[goodmask[:],:]  # remove bad rows
    
    #print("[row-tranform] (post mask) coord: ",coord.shape)    
    sa = sa_values[ coord[:,0], coord[:,1], coord[:,2], : ]
    #print("[row-transform] (post mask) sa: ",sa.shape)

    # normalize charge features to keep within range
    feat /= 10000.0

    # normalize pe features as well
    #pe = np.log((row['flashpe']+1.0e-4)/100.0)
    pe = row['flashpe']/1000.0+1.0e-8

    #print(row)
    result = {"coord":coord,
              "feat":feat,
              "sa":sa,
              "flashpe":pe,
              "event":row["event"],
              "matchindex":row["matchindex"]}

    #print("[ran default_row_transform]")
    return result

default_transform_spec = TransformSpec(_default_transform_row,
                                       removed_fields=['sourcefile','run','subrun','ancestorid'],
                                       edit_fields=[('sa',np.float32,(None,32),NdarrayCodec(),False)])

def flashmatchdata_collate_fn(datalist):
    """
    data will be a list.
    fields we expect and how we will package them
    coord: list of torch tensors (ready for sparse matrix)
    feat: list of torch tensors (ready for sparse matrix)
    sa: list of torch tensors (ready for sparse matrix)
    flashpe: return single 2d torch tensor of (B,32)
    event: return single 1d torch tensor (B)
    matchindex: return single 1d torch tensor(B)
    """

    #print("[collate_fn] datalist")
    #print("len(datalist)=",len(datalist))
    #print(datalist)
    
    #output dictionary with fields
    batchsize=len(datalist)
    collated = {"coord":[],
                "feat":[],
                "flashpe":np.zeros( (batchsize,32), dtype=np.float32 ),
                "event":np.zeros( (batchsize), dtype=np.int64 ),
                "matchindex":np.zeros( (batchsize), dtype=np.int64 ),
                "batchentries":np.zeros( (batchsize), dtype=np.int64),
                "batchstart":np.zeros((batchsize),dtype=np.int64)}

    startindex = 0
    #print("batch start index: ",startindex)
    for i in range(batchsize):
        row = datalist[i]
        collated["coord"].append( row["coord"] )
        collated["feat"].append( np.concatenate((row["feat"],row["sa"]),axis=1) )
        collated["event"][i] = row["event"]
        collated["matchindex"][i] = row["matchindex"]
        collated["flashpe"][i,:] = row["flashpe"][:]
        collated["batchentries"][i] = row["coord"].shape[0]
        collated["batchstart"][i] = startindex
        startindex += row["coord"].shape[0]
    #print("batch end index: ",startindex)

    coords, feats = me.utils.sparse_collate( coords=collated["coord"], feats=collated["feat"] )
    #print("collated coords: ",coords)
    #print("collated feats: ",feats)

    collated["coord"] = coords
    collated["feat"] = feats

    #print("collate ran")
    
    return collated
        
def make_dataloader( dataset_folder, num_epochs, shuffle_rows, batch_size,
                     row_transformer=None,
                     seed=1,
                     workers_count=1,
                     worker_batchsize=1,
                     removed_fields=['sourcefile','run','subrun','ancestorid']):
    
    if not row_transformer:
        # use default
        transform_spec = default_transform_spec
    else:
        transform_func = row_transformer
        transform_spec = TransformSpec(transform_func, removed_fields=removed_fields)        

    loader =  DataLoader( make_reader(dataset_folder, num_epochs=num_epochs,
                                      transform_spec=transform_spec,                                      
                                      seed=seed,
                                      workers_count=workers_count,
                                      shuffle_rows=shuffle_rows ),
                          batch_size=batch_size,
                          collate_fn=flashmatchdata_collate_fn)
    return loader

def get_rows_from_data_iterator( data_iter, verbose=False ):
    """
    ask the configured petastorm loader to return the next iteration.
    we return a list of individual rows return.
    somtimes will return more than one row if multiple-workers used.
    """
    ntries = 0
    column_dict = {}

    try:
        rowset = next(data_iter)
    except:
        print("iterator exhausted. reset data loader and retry")
        return None

    nrows_returned = rowset['coord'].shape[0]
    if verbose: print("[flashmatchdata.get_rows_from_data_iterator] nrows_ret=",nrows_returned)
    for k in rowset:
        column_dict[k] = []
        if verbose:
            print("  processing column=",k," type=",type(rowset[k]))
        if type(rowset[k]) is torch.Tensor:
            # break into list of tensors. easier for sparsetensor preprocessing.
            for irow in range(nrows_returned):            
                column_dict[k].append(rowset[k][irow])
        elif type(rowset[k]) is list:
            column_dict[k] = rowset[k]
        elif type(rowset[k]) is tuple:
            rowdata[k] = rowset[k]
            
    return column_dict
    

    
    
if __name__ == "__main__":
    
    # this is used for testing and debugging
    # also provides an example of how to use the data loader
    import time

    DATAFOLDER='file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/flashmatch_mc_data'
    NUM_EPOCHS=1
    WORKERS_COUNT=4
    WORKER_BATCH_SIZE=2
    BATCH_SIZE=64

    dataloader = make_dataloader( DATAFOLDER, NUM_EPOCHS, True, BATCH_SIZE,
                                  workers_count=WORKERS_COUNT)
    data_iter = iter(dataloader)

    NITERS = 10
    tstart = time.time()
    for ii in range(NITERS):
        row = next(data_iter)
        print("[ITER ",ii,"] ====================")
        print("row keys: ",row.keys())
        print("coord: ",row["coord"].shape)
        print("event: ",row["event"])
        print("matchindex: ",row["matchindex"])
        print("entries per batch: ",row["batchentries"])
    tend = time.time()
    print("===========")
    print("time per batch [batchsize=",BATCH_SIZE,"]: ",(tend-tstart)/float(NITERS)," secs/batch")
          
    
    

    
