import torch


def calc_qmean_and_qweighted_pos( coord_batch, voxel_q_batch,
                                  batch_starts, batch_ends,
                                  vox_len_cm=5.0 ):
    """
    coord_batch: expect (N,3) torch.int64 tensor
    voxel_q_batch: expect (N,3) torch.float32 tensor
    batch_starts: expect (B,) torch.int64 tensor
    batch_ends: expect (B,) torch.int64 tensor
    """
    batchsize = batch_starts.shape[0]

    vox_qmean = voxel_q_batch.mean( dim=1 ) # reduces (N,3) to (N,)

    pos_mean = torch.zeros( (batchsize,3), device=voxel_q_batch.device, dtype=voxel_q_batch.dtype )

    qmean_sum = torch.zeros( batchsize, device=voxel_q_batch.device, dtype=torch.float32 )

    vox_pos = coord_batch[:,:].to(torch.float32)*vox_len_cm # cast means, should be a copy

    for ib in range(batchsize):

        pos_batch = vox_pos[batch_starts[ib]:batch_ends[ib]] # (N_i, 3), view of vox_pos elements
        q_batch = vox_qmean[batch_starts[ib]:batch_ends[ib]] # (N_i,)
        qmean_sum[ib] = q_batch.sum()

        pos_batch_T = torch.transpose( pos_batch, 1, 0 ) # transpose for broadcast (3, N) * (N,), in-place
        pos_batch_T *= q_batch
        #torch.transpose( pos_batch, 1, 0 ) *= q_batch  
        
        pos_mean[ib] = pos_batch.sum()/qmean_sum[ib]

    return pos_mean, qmean_sum
    
