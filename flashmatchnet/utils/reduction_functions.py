import torch


def calc_qmean_and_qweighted_pos( coord_batch, 
                                  voxel_q_batch,
                                  mask,
                                  vox_len_cm=5.0 ):
    """
    coord_batch: expect (B,N,3) torch.int64 tensor
    voxel_q_batch: expect (B,N,3) torch.float32 tensor
    mask (B,N,1)
    batch_starts: expect (B,) torch.int64 tensor
    batch_ends: expect (B,) torch.int64 tensor
    """
    batchsize, Nvoxels, Ndim = coord_batch.shape
    #print(batchsize," ",Nvoxels," ",Ndim)

    vox_qmean = voxel_q_batch.mean( dim=2 ) # reduces (B,N,3) to (B,N)
    #print('vox_qmean: ',vox_qmean.shape)

    #vox_pos = coord_batch[:,:].to(torch.float32) # cast means, should be a copy

    q_batch = vox_qmean*mask # (B,N)*(B,N)

    pos_batch = coord_batch*q_batch.reshape((batchsize,Nvoxels,1)) # (B,N,3)*(B,N,1)
        
    qmean_sum = q_batch.sum(dim=1) # (B,N) to (B,)

    pos_mean = pos_batch.sum(dim=1)/qmean_sum.reshape((batchsize,1)) # (B,3)/(B,1)

    return pos_mean, qmean_sum
    
