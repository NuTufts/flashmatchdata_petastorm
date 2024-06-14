import os,sys
import torch
import numpy as np
import MinkowskiEngine as ME

def mixup( batch, device, factor_range=[0.5,1.5] ):
    """
    make linear combination of neighboring batch entries.
    this compositionality *should* work
    """
    batchsize = len(batch['batchentries'])
    nbatchout = int(batchsize/2)
    if batchsize%2!=0:
        nbatchout += 1

    out = {"matchindex":[],
           "event":[],
           "coord":[],
           "feat":[],
           "flashpe":np.zeros( (nbatchout,32), dtype=np.float32 ),
           "batchentries":np.zeros( (nbatchout), dtype=np.int32 ),
           "batchstart":np.zeros((nbatchout),dtype=np.int32),
           "batchend":np.zeros((nbatchout),dtype=np.int32)}
    
              
    scale = np.random.uniform( factor_range[0], factor_range[1], size=batchsize )

    istart = 0
    for i in range(nbatchout):

        if i+1>=nbatchout and batchsize%2!=0:
            # odd batchsize, so no pair
            s = batch['batchstart'][2*i]
            e = batch['batchend'][2*i]
            out["coord"].append( torch.from_numpy(batch['coord'][s:e,:]) )
            out['feat'].append( torch.from_numpy(batch['feat'][s:e,:]) )
            out['flashpe'][i,:] = batch['flashpe'][2*i,:]
        else:
            
            # combine 2*i and 2*i+1
            s1 = batch['batchstart'][2*i]
            e1 = batch['batchend'][2*i]

            s2 = batch['batchstart'][2*i+1]
            e2 = batch['batchend'][2*i+1]
            
            A = batch['coord'][s1:e1,:] #.astype(np.float32)
            B = batch['coord'][s2:e2,:] #.astype(np.float32)
            fA = batch['feat'][s1:e1,:]*scale[2*i]
            fB = batch['feat'][s2:e2,:]*scale[2*i+1]
            combo_coord = np.concatenate( (A,B), axis=0 )
            combo_feat  = np.concatenate( (fA,fB), axis=0 )
            #print("combo_coord: ",combo_coord.shape)
            #print("combo_feat: ",combo_feat.shape)
            Norig = A.shape[0]+B.shape[0]
            
            C = ME.SparseTensor( features=torch.from_numpy(combo_feat).to(device),
                                 coordinates=torch.from_numpy(combo_coord).to(device),
                                 quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM )
            #C = ME.sparse_collate( 
            #print(C.coordinates_at(batch_index=0).shape)
            #print(C.features_at(batch_index=0).shape)
            #print(C.C.shape)
            #print(C.F.shape)
            Ncombine = C.C.shape[0]
            #print("Norig=",Norig," --> Ncombine=",Ncombine)
            #print("A.shape=",A.shape," + B.shape=",B.shape," ---> Ncombined.shape=",C.C.shape)
            if Ncombine<Norig:
                print("overlap combined: Norig=",Norig," --> Ncombine=",Ncombine)
            #    #print(C.F[:10,0])
            #    #print(fA[:10,0]+fB[:10,0])
                
            
            out['coord'].append( C.C[:,1:].int() )
            out['feat'].append( C.F )
            out['flashpe'][i,:] = batch['flashpe'][2*i,:]*scale[2*i] + batch['flashpe'][2*i+1,:]*scale[2*i+1]
        out['matchindex'].append( batch['matchindex'][2*i] )
        out["event"].append( batch['event'][2*i] )
        out['batchentries'][i] = C.C.shape[0]
        out['batchstart'][i] = istart
        out['batchend'][i] = istart+C.C.shape[0]
        istart = istart+C.C.shape[0]
        

    coords, feats = ME.utils.sparse_collate( coords=out["coord"], feats=out["feat"] )
    out['coord'] = coords
    out['feat']  = feats
    #out['flashpe'] = torch.from_numpy(out['flashpe']).to(device)

    if "run" in batch:
        out["run"] = [ batch['run'][2*i] for i in range(nbatchout) ]
    if "subrun" in batch:
        out["subrun"] = [ batch['subrun'][2*i] for i in range(nbatchout) ]
    
            
    return out


def scale_small_charge( batch, x_threshold_cm=175.0, pesum_limit=1.0, scale_factor_max=2.0):
    """
    To help learn the visibility function near the cathode, we scale
    up small clusters.

    We measure the total pe and the q-weighted x-position.
    """

    entries_per_batch = batch['batchentries']
    batchsize = entries_per_batch.shape[0]
    
    start_per_batch = batch['batchstart']
    end_per_batch   = batch['batchend']
    
    scale = 1.0+np.random.uniform( 0.0, scale_factor_max, size=batchsize )

    print("TEST!! batch['flashpe'] is: ", batch['flashpe'])
    print("TEST!! batch['flashpe'].shape is: ", batch['flashpe'].shape)
    print("TEST!! batch['flashpe'] type is: ", type(batch['flashpe']))

    batch['flashpe'] = batch['flashpe'].numpy()
    
    pesum = np.sum(  batch['flashpe'], axis=1 )
    for b in range(batchsize):
        if pesum[b]>pesum_limit:
            continue
        # calc q-weighted average
        s = start_per_batch[b]
        e = end_per_batch[b]
        x = batch['coord'][s:e,0] # batch index is first index
        q_per_plane = np.average( batch['feat'][s:e,:3], axis=1 )
        #print("small pesum, calc q-weighted mean x")
        #print(x.shape)
        #print(q_per_plane.shape)

        xq = x*q_per_plane*5.0 # change coord to 5.0 cm
        #xq_sum = xq[ start_per_batch[b]:end_per_batch[b] ].sum()
        #q_sum  = q_per_plane[ start_per_batch[b]:end_per_batch[b] ].sum()
        xq_sum = xq.sum()
        q_sum  = q_per_plane.sum()
        x_qweighted = xq_sum/q_sum
        if x_qweighted>x_threshold_cm:
            # far candidate
            #print("far candidate: pesum=",pesum[b]," and x=",x_qweighted)
            #print("  scale up q and pe by ",scale[b])
        #batch['feat'][b][ start_per_batch[b]:end_per_batch[b],:] *= scale[b]
            batch['feat'][s:e,:] *= scale[b]
            batch['flashpe'][b,:] *= scale[b]

    return batch
    
