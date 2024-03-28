import os,sys

import numpy as np
import torch

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

import MinkowskiEngine as me

from .petastormschema import FlashMatchSchema

def _default_transform_row( row ):
    #print(row)
    # original tensors from database
    coord = row['coord']
    feat  = row['feat']
    #print("[row-tranform] (pre-max index mask) coord: ",coord.shape)
    
    goodmask_i = coord[:,0]<54
    goodmask_j = coord[:,1]<49
    goodmask_k = coord[:,2]<210
    goodmask = goodmask_i*goodmask_j*goodmask_k

    #print("[row-transform] num_good=",goodmask.sum())

    coord = coord[goodmask[:],:] # remove bad rows
    feat  = feat[goodmask[:],:]  # remove bad rows
    
    # normalize charge features to keep within range
    feat /= 10000.0

    # normalize pe features as well
    #pe = np.log((row['flashpe']+1.0e-4)/100.0)
    pe = row['flashpe']/1000.0+1.0e-4

    #print(row)
    result = {"coord":coord,
              "feat":feat,
              "flashpe":pe,
              "event":row["event"],
              "matchindex":row["matchindex"]}

    #print("[ran default_row_transform]")
    return result

default_transform_spec = TransformSpec(_default_transform_row,
                                       removed_fields=['sourcefile','run','subrun','ancestorid'])

default_transform_spec_with_sa = TransformSpec(_default_transform_row,
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
                "batchstart":np.zeros((batchsize),dtype=np.int64),
                "batchend":np.zeros((batchsize),dtype=np.int64)}

    startindex = 0
    #print("batch start index: ",startindex)
    for i in range(batchsize):
        row = datalist[i]
        collated["coord"].append( row["coord"] )
        collated["feat"].append( row["feat"] )
        collated["event"][i] = row["event"]
        collated["matchindex"][i] = row["matchindex"]
        collated["flashpe"][i,:] = row["flashpe"][:]
        collated["batchentries"][i] = row["coord"].shape[0]
        collated["batchstart"][i] = startindex
        collated["batchend"][i] = startindex+row["coord"].shape[0]
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
                     custom_collate_fn=None,
                     seed=1,
                     workers_count=1,
                     worker_batchsize=1,
                     removed_fields=['sourcefile','run','subrun','ancestorid'],
                     edit_fields=[]):
    
    if not row_transformer:
        # use default
        transform_spec = default_transform_spec
    else:
        transform_func = row_transformer
        transform_spec = TransformSpec(transform_func,
                                       removed_fields=removed_fields,
                                       edit_fields=edit_fields)

    if custom_collate_fn is None:
        # use default
        custom_collate_fn = flashmatchdata_collate_fn

    loader =  DataLoader( make_reader(dataset_folder, num_epochs=num_epochs,
                                      transform_spec=transform_spec,                                      
                                      seed=seed,
                                      workers_count=workers_count,
                                      shuffle_rows=shuffle_rows ),
                          batch_size=batch_size,
                          collate_fn=custom_collate_fn)
    return loader



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
    
    print("===================================")
    print("time per batch [batchsize=",BATCH_SIZE,"]: ",(tend-tstart)/float(NITERS)," secs/batch")
    print("===================================")
    
