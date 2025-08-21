import os,sys
import torch
import torch.nn as nn
import wandb

import MinkowskiEngine as ME

from .reduction_functions import calc_qmean_and_qweighted_pos
from .coord_and_embed_functions import prepare_mlp_input_embeddings,prepare_mlp_input_variables

def validation_calculations( valid_batch,
                             net,
                             valid_loss_fn,
                             batchsize,
                             device,
                             pmtpos,
                             nvalid_iters=100,
                             use_embed_inputs=True):
    """
    calculate various metrics for monitor training progress
    passing mlp is a kluge. I need to separate out embedding functions used in 
       prepare_mlp_input_embeddings()
    """

    net.eval()
    
    table_data = []

    #mean_pos_v = []
    #pe_target_sum_v = []
    #pe_pred_sum_v = []
    #qmean_sum_v = []

    floss_tot_ave = 0.0
    floss_emd_ave = 0.0
    floss_mag_ave = 0.0
    
    # target pe (N,32)
    vcoord = valid_batch['avepos'].to(device)
    Nb,Nv,Ndim = vcoord.shape
    pe_target = valid_batch['observed_pe_per_pmt_normalized'].to(device)
    
    # record true values for tracking metrics
    pe_sum_target = pe_target.sum(dim=1)
    pe_max_target, pe_max_target_idx = pe_target.max(1)
        
    q_feat = valid_batch['planecharge_normalized'].to(device)
    mask   = valid_batch['mask'].to(device)

    qweighted_pos, qmean_sum = calc_qmean_and_qweighted_pos( vcoord, q_feat, mask )

    # prepare inputs to net
    if use_embed_inputs:
        if mlp is not None:
            vox_feat, q_per_pmt = prepare_mlp_input_embeddings( vcoord, q_feat, mlp )
        else:
            vox_feat, q_per_pmt = prepare_mlp_input_embeddings( vcoord, q_feat, net )
    else:
        vox_feat = prepare_mlp_input_variables( vcoord.reshape(-1,3), q_feat.reshape(-1,3), pmtpos, vox_len_cm=1.0 )
    # reshape to send all voxels through network at once
    #print("[validation calculations] vox_feat.shape=",vox_feat.shape)
    N,C,K = vox_feat.shape
    vox_feat = vox_feat.reshape( (N*C,K) )
    q = vox_feat[:,-1:]
    vox_feat = vox_feat[:,:-1]
    K += -1

    #input = ME.SparseTensor(features=q_feat, coordinates=vcoord)

    # forward pass
    #pmtpe_per_voxel = net(vox_feat_nc, q_nc).reshape( (N,C) )
    pmtpe_per_voxel = net(vox_feat,q)
    pmtpe_per_voxel = pmtpe_per_voxel.reshape( (Nb,Nv,C))

    pmtpe_per_voxel = mask.reshape((Nb,Nv,1))*pmtpe_per_voxel
    pmtpe_per_voxel = pmtpe_per_voxel.sum(dim=1)

    # loss
    loss_tot,(floss_tot,floss_emd,floss_mag,pred_pesum,pred_pemax) = valid_loss_fn( pmtpe_per_voxel,
                                                                                    pe_target,
                                                                                    None, None, mask=mask )

    # save table data
    for ib in range(batchsize):
            
        table_data.append( [ qweighted_pos[ib][0].cpu().item(), # x
                             qweighted_pos[ib][1].cpu().item(), # y
                             qweighted_pos[ib][2].cpu().item(), # z
                             qmean_sum[ib].cpu().item(),     # qmean                                 
                             pe_sum_target[ib].cpu().item(), # target pe sum
                             pe_max_target[ib].cpu().item(), # target pe max 
                             pred_pesum[ib].cpu().item(),    # pred pe sum
                             pred_pemax[ib].cpu().item() ] ) # pred pe max
            
    floss_tot_ave += floss_tot
    floss_emd_ave += floss_emd
    floss_mag_ave += floss_mag

    fnexamples = float(nvalid_iters)
    floss_tot_ave /= fnexamples
    floss_emd_ave /= fnexamples
    floss_mag_ave /= fnexamples

    # make tables of data for wandb plot
    #wdb_table = wandb.Table(data=table_data, columns = ["x", "y","z",
    #                                                    "qmean",
    #                                                    "pe_sum_target",
    #                                                    "pe_max_target",
    #                                                    "pe_sum_pred",
    #                                                    "pe_max_pred"])
    #return {"loss_tot_ave":floss_tot_ave,
    #        "loss_emd_ave":floss_emd_ave,
    #        "loss_mag_ave":floss_mag_ave,
    #        "table_data":wdb_table}
    return {"loss_tot_ave":floss_tot_ave,
            "loss_emd_ave":floss_emd_ave,
            "loss_mag_ave":floss_mag_ave}
        
        
