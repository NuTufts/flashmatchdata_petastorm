import os, sys, time
from typing import Dict, Any
import torch
from flashmatchnet.utils.pmtpos import create_pmtpos_tensor
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings, prepare_mlp_input_variables
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor


def apply_normalization(batch: Dict[str, torch.Tensor], 
                       config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Apply normalization to batch data based on config"""
    
    try:
        norm_config = config['dataloader']
    except:
        raise ValueError('Normalization config parameters expected in "dataloader" block in config dict')
    
    # Normalize PMT values if specified
    if 'pmt_norm_params' in norm_config:
        pmt_params = norm_config['pmt_norm_params']
        offset = pmt_params['offset']
        scale = pmt_params['scale']
        
        # Apply log transform and normalization to observed PE
        if 'observed_pe_per_pmt' in batch:
            if pmt_params['transform']=='log':
                log_pe = torch.log(1.0 + batch['observed_pe_per_pmt'])
                batch['observed_pe_per_pmt_normalized'] = (log_pe + offset) / scale
            elif pmt_params['transform']=='linear':
                batch['observed_pe_per_pmt_normalized'] =  (batch['observed_pe_per_pmt']+offset)/scale
            else:
                raise ValueError("observed PE transform option not recognized")
    
    # Normalize plane charge if specified
    if 'planecharge_norm_params' in norm_config:
        charge_params = norm_config['planecharge_norm_params']
        offsets = torch.tensor(charge_params['offset'], device=batch['planecharge'].device)
        scales = torch.tensor(charge_params['scale'], device=batch['planecharge'].device)
        
        # Apply log transform and normalization to plane charge
        if 'planecharge' in batch:
            if charge_params.get('transform')=='log':
                log_charge = torch.log(1.0 + batch['planecharge'])
                batch['planecharge_normalized'] = (log_charge + offsets) / scales
            elif charge_params.get('transform')=='linear':
                batch['planecharge_normalized'] = (batch['planecharge']+offsets)/scales
            else:
                raise ValueError("plane charge transform option not recognized")
    
    return batch

def prepare_batch_input( batch, config, device, pmtpos=None):
    """
    we have to do a few things besides just passing in the 
    array of charge voxel

    inputs:
    batch: dictionary containing batch tensors.  must have
      'avepos': (Nbatchsize,Nvoxels,3)
      'planecharge': (Nbatchsize,Nvoxels,Nplanes)
      'mask': (Nbatchsize,Nvoxels) [optional: indicates which pixels are valid (1) or just adding (0)]
      'observed_pe_per_pmt':(Nbatchsize,Npmts) [optional: training target]
    config: dictionary with config parameters. parameter blocks expected:
      'dataloader': config parameters for data loader
      'model': config parameters model

    """

    tstart_dataprep = time.time()

    if pmtpos is None:
        pmtpos = create_pmtpos_tensor()

    batch = apply_normalization(batch,config)

    coord  = batch['avepos'].to(device)
    Nb,Nv,Nd = coord.shape
    

    data_config  = config['dataloader']
    model_config = config['model']
    debug_mode = data_config.get('debug',False)

    q_feat = batch['planecharge_normalized'].to(device)
    mask   = batch['mask'].to(device)
    n_voxels = batch['n_voxels'].to(device)
    start_per_batch = torch.zeros( config["dataloader"].get('batchsize'), dtype=torch.int64 ).to(device)
    end_per_batch   = torch.zeros( config["dataloader"].get('batchsize'), dtype=torch.int64 ).to(device)
    for i in range(config['dataloader'].get('batchsize')):
        start_per_batch[i] = i*Nv
        end_per_batch[i]   = (i+1)*Nv

    if debug_mode:
        print("-"*80)
        print("Coordinate shape: Nb, Nv, Nd: ",Nb," ",Nv," ",Nd)
        print('n_voxels: ',n_voxels)
        print('coord: ',coord.shape)
        print('q_feat: ',q_feat.shape)
        print('start_per_batch: ',start_per_batch.shape)
        print('mask: ',mask.shape)
        print('num_ones: ',(mask==1).sum())
        print('num_zeros: ',(mask==0).sum())
        print('start_per_batch: ',start_per_batch)
    
    # for each coord, we produce the other features
    if model_config.get('use_cos_input_embedding_vectors'):
        vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, pmtpos, vox_len_cm=1.0 )
    else:
        # takes in (N,3) coord and (N,3) charge tensor
        # before coord and q_feat go in,
        # they get reshaped from (Nb,Nv,3) -> (Nb*Nv,3)
        vox_feat = prepare_mlp_input_variables( coord.reshape(-1,3), q_feat.reshape(-1,3), pmtpos, vox_len_cm=1.0 )
        #print('vox feat: ',vox_feat.shape,' device=',vox_feat.device)
        
    dt_dataprep = time.time()-tstart_dataprep
    tstart_forward = time.time()
    Nbv,Npmt,K = vox_feat.shape
    vox_feat = vox_feat.reshape( (Nbv*Npmt,K) )
    q = vox_feat[:,-1:]
    mask = torch.repeat_interleave( mask.reshape(-1,1), Npmt, dim=0 ).reshape( (Nbv,Npmt,1) )
    vox_feat = vox_feat[:,:-1]
    K += -1

    if debug_mode:
        print("INPUTS ==================")
        print("vox_feat.shape=",vox_feat.shape," from (Nb x Nv,Npmt,Nk)=",(Nbv,Npmt,K))
        print("q.shape=",q.shape)

    return vox_feat, q, mask
