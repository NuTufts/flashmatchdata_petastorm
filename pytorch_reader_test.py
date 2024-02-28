import os,sys
import numpy as np
import torch
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader
from petastorm.codecs import NdarrayCodec

torch.manual_seed(1)

#dataset_folder = 'file:///tmp/test_flash_dataset'
dataset_folder = 'file:///cluster/tufts/wongjiradlabnu/twongj01/dev_petastorm/datasets/test_flash_dataset'

from sa_table import load_satable_fromnpz,get_satable_maxindices

sa_coord, sa_values = load_satable_fromnpz()
sa_maxindices = get_satable_maxindices()

def _transform_row( row ):
    
    

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

    #print(row)
    result = {"coord":coord,
              "feat":feat,
              "sa":sa,
              "flashpe":row["flashpe"],              
              "event":row["event"],
              "matchindex":row["matchindex"]}
    
    return result

transform = TransformSpec(_transform_row,
                          removed_fields=['sourcefile','run','subrun','ancestorid'],
                          edit_fields=[('sa',np.float32,(None,32),NdarrayCodec(),False)])
#transform = TransformSpec(_transform_row, removed_fields=[])
#reader = make_reader( dataset_folder, num_epochs=1, transform_spec=transform, seed=1, shuffle_rows=False )
#for row in reader:
#    print(row)


strides = get_satable_maxindices()
        
with DataLoader( make_reader(dataset_folder, num_epochs=1,
                             transform_spec=transform,
                             workers_count=4,
                             seed=1, shuffle_rows=True ),
                 batch_size=1 ) as loader:

    batchsize = 16
    niters = 500

    iterator = iter(loader)

    for itrain in range(niters):

        coord_list = []
        feat_list  = []
        
        print("TRAIN ITER[",itrain,"] =============")
        ntries = 0
        while len(coord_list)<batchsize:
            try:
                row = next(iterator)
            except:
                print("iterator exhausted. reset")
                iterator = iter(loader)
                row = next(iterator)
        
            print(" [ncall ",ntries,"] ==================")
            print(" event: ",row['event']," matchindex: ",row['matchindex'])
            print(" coord: ",row['coord'].shape)
            print(" feat: ",row['feat'].shape)
            print(" flashpe: ",row['flashpe'].shape)

            n_worker_return = row['coord'].shape[0]
            coord = row['coord']
            feat  = row['feat']

            for i in range(n_worker_return):
                if coord.shape[1]>0:
                    coord_list.append( coord[i,:,:] )
                    feat_list.append( feat[i,:,:] )
            ntries += 1
            if ntries>100:
                print("infinite loop trying to fill nonzero charge tensors")
                sys.exit(1)
        

        print("num coord tensors ready for training iteration: ",len(coord_list))
        assert(len(coord_list)==batchsize) # check that we have the right number of tensors

        # make sparse tensor here
        #sparse_tensor_input = me.sparse_collate( coord=coord_list, feat=feat_list )

        # input network

        # sa multiply
    
        # # spot check code
        # if row['coord'].shape[1]==0:
        #     print("zero entry! BAD")
        #     continue
        
        # x = row['coord'][0,:3,:]
        # print("x: ",x.shape)
        # for ii in range(3):

        #     r = x[ii,:]
        #     print(r)
        #     txtrow = r[0]*strides[2]*strides[1] + r[1]*strides[2] + r[2]
        #     print("spot check [",ii,"]==================")
        #     print(" coords: ",x[ii,:],"  textrow=",txtrow)
        #     print(" sa: ",row['sa'][0,ii,:3])
        
        # if True:
        #     break
