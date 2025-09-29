import os,sys,time
import ROOT as rt
import torch
import torch.nn as nn
from array import array
import yaml
from typing import Dict, Any, Optional
from math import fabs

import torch
from torch.utils.data import Dataset, DataLoader

import geomloss

# HDF5 datasets
from flashmatchnet.data.read_flashmatch_hdf5 import FlashMatchVoxelDataset
from flashmatchnet.data.flashmatch_mixup import MixUpFlashMatchDataset
from flashmatchnet.utils.pmtpos import getPMTPosByOpDet
from flashmatchnet.utils.load_model import load_model

# Input embeddings
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings, prepare_mlp_input_variables
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor



"""
run_siren_inference.py

We use this script to study the performance of the model. 
Our goal is to run on the validation data set rather than on provide a module one can run in the ntuple-maker or in the Reco.

We need to:
1. configure the model via a yaml config
2. also via the yaml, we need to setup a data reader for the validation data
3. we run the inference, saving prediction
4. we have the light model prediction in the validation data set, 
   so we can compare via pe total and the sinkhorn divergence loss.
   so we should save the difference in terms of a change in MSE for the totl
   and a change in the sinkhorn divergence (positive or negative)
"""

import flashmatchnet

def create_data_loaders(config: Dict[str, Any]) -> (DataLoader, Dataset):
    """Create training and validation data loaders using the new HDF5 dataset"""
    
    dataloader_config = config['dataloader']
    
    print(f"Loading validation data from: {dataloader_config['valid_filelist']}")
    valid_base_dataset = FlashMatchVoxelDataset(
        hdf5_files=dataloader_config['valid_filelist'],
        max_voxels=dataloader_config.get('max_voxels', 500),
        load_to_memory=False
    )
    
    # Apply MixUp augmentation if specified
    use_mixup  = dataloader_config.get('use_mixup',True)
    mixup_prob = dataloader_config.get('mixup_prob')
    
    if use_mixup and mixup_prob > 0:
        print(f"Applying MixUp augmentation with probability {mixup_prob}")
        
        # Create MixUp datasets
        valid_dataset = MixUpFlashMatchDataset(
            base_dataset=valid_base_dataset,
            mixup_prob=mixup_prob,
            alpha=dataloader_config.get('mixup_alpha', 1.0),
            max_total_voxels=dataloader_config.get('max_voxels', 500) * 2
        )
        
        # Use custom collate function for MixUp
        from flashmatchnet.data.flashmatch_mixup import mixup_collate_fn
        collate_fn = mixup_collate_fn
    else:
        valid_dataset = valid_base_dataset
        collate_fn = None
    
    # Create data loaders 
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=dataloader_config['batchsize'],
        shuffle=False,  # Don't shuffle validation
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config.get('pin_memory', False),
        drop_last=True
    )
    
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Batch size: {dataloader_config['batchsize']}")
    
    return valid_dataloader, valid_dataset

def apply_normalization(batch: Dict[str, torch.Tensor], 
                       config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Apply normalization to batch data based on config"""
    
    norm_config = config['dataloader']
    
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
    
def undo_pmt_normalization(pe_per_pmt, config):
    pmt_params = config.get('dataloader').get('pmt_norm_params')
    if pmt_params['transform']=='linear':
        offset = pmt_params['offset']
        scale = pmt_params['scale']
        pe_per_pmt_denorm = scale*pe_per_pmt-offset
    
    return pe_per_pmt_denorm

def prepare_input( batch, config, pmtpos, device ):
    """
    we have to do a few things besides just passing in the 
    array of charge voxels.
    """

    tstart_dataprep = time.time()

    device = torch.device( config['inference'].get('device'))
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

def create_pmtpos(apply_y_offset=True):
    # copy position data into numpy array format
    pmtpos = torch.zeros( (32, 3) )
    for i in range(32):
        opdetpos = getPMTPosByOpDet(i,use_v4_geom=True)
        for j in range(3):
            pmtpos[i,j] = opdetpos[j]
    # change coordinate system to 'tensor' system
    # main difference is y=0 is at bottom of TPC 
    if apply_y_offset:       
        pmtpos[:,1] -= -117.0
    # The pmt x-positions are wrong (!).
    # They would be in the TPC with the values I have stored.
    # So move them outside the TPC
    pmtpos[:,0] = -20.0
    # now corrected to be at -11, but need to keep things consistent
    return pmtpos

def makehists( histname_stem, pred_pe_per_pmt, obs_pe_per_pmt, ubmodel_pe_per_pmt ):
    import ROOT as rt

    hists = {
        'siren':rt.TH1D(f"{histname_stem}_siren","",32,0,32),
        'obs':rt.TH1D(f"{histname_stem}_obs","",32,0,32),
        'ub':rt.TH1D(f"{histname_stem}_ubmodel","",32,0,32)
    }
    for htype,h in hists.items():
        h.SetLineWidth(2)
    hists['siren'].SetLineColor(rt.kRed)
    hists['obs'].SetLineColor(rt.kBlack)
    hists['ub'].SetLineColor(rt.kBlue)
    for ipmt in range(32):
        hists['siren'].SetBinContent( ipmt+1, pred_pe_per_pmt[ipmt] )
        hists['obs'].SetBinContent( ipmt+1, obs_pe_per_pmt[ipmt] )
        hists['ub'].SetBinContent( ipmt+1, ubmodel_pe_per_pmt[ipmt] )

    return hists

def calc_sinkhorn_divergences( pred_pe_per_pmt, obs_pe_per_pmt, ubmodel_pe_per_pmt ):
    pass

def main(config_path):

    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    debug = config['inference'].get('debug',False)
    NENTRIES = config['inference'].get('num_max_entries',-1)
    if NENTRIES<0:
        num_batches = -1
    else:
        num_batches = NENTRIES/config['dataloader'].get('batchsize')+1

    device = torch.device( config['inference'].get('device'))
    return_fvis = config['model']['lightmodelsiren'].get('return_fvis',False)

    sinkhorn_fn = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=0.05).to(device)

    # we make the x and y tensors
    x_pred   = get_2d_zy_pmtpos_tensor(scaled=True).to(device) # (32,2)
    y_target = get_2d_zy_pmtpos_tensor(scaled=True).to(device) # (32,2)

    dataloader, dataset = create_data_loaders(config)

    print("Loaded DataLoader. Num entries: ",len(dataset))

    siren = load_model(config)
    siren = siren.to(device)
    siren.eval()

    print("Loaded SIREN Model")
    print("="*80)
    print(siren)
    print("="*80)

    pmtpos = create_pmtpos(apply_y_offset=config['dataloader'].get('apply_pmtpos_yoffset',False)).to(device)
    print("PMTPOS: ",pmtpos.shape)
    Npmt,dk = pmtpos.shape

    import ROOT as rt
    from ROOT import std
    rt.gStyle.SetOptStat(0)

    if debug:
        c = rt.TCanvas("c","c",2400,2400)
        c.Divide(4,4)

    # create output 
    fout_name = config['inference'].get('output_filename','output_temp_siren_inference.root')
    fout = rt.TFile( fout_name, 'recreate' )
    tree = rt.TTree('siren_inference','Siren Inference Analysis Variables')

    siren_pe_per_pmt_v = std.vector('double')(Npmt,0)
    obs_pe_per_pmt_v   = std.vector('double')(Npmt,0)
    ub_pe_per_pmt_v    = std.vector('double')(Npmt,0)
    siren_pe_tot       = array('f',[0.0])
    obs_pe_tot         = array('f',[0.0])
    ub_pe_tot          = array('f',[0.0])
    siren_sinkhorn     = array('f',[0.0])
    ub_sinkhorn        = array('f',[0.0])
    siren_fracerr      = array('f',[0.0])
    ub_fracerr         = array('f',[0.0])
    fvis_min           = array('f',[0.0])
    fvis_max           = array('f',[0.0])
    fvis_mean          = array('f',[0.0])

    tree.Branch('siren_pe_per_pmt',  siren_pe_per_pmt_v)
    tree.Branch('obs_pe_per_pmt',    obs_pe_per_pmt_v)
    tree.Branch('ub_pe_per_pmt',     ub_pe_per_pmt_v)
    tree.Branch('siren_pe_tot',      siren_pe_tot,       'siren_pe_tot/F')
    tree.Branch('obs_pe_tot',        obs_pe_tot,         'obs_pe_tot/F')
    tree.Branch('ub_pe_tot',         ub_pe_tot,          'ub_pe_tot/F')
    tree.Branch('siren_sinkhorn',    siren_sinkhorn,     'siren_sinkhorn/F')
    tree.Branch('ub_sinkhorn',       ub_sinkhorn,        'ub_sinkhorn/F')
    tree.Branch('siren_fracerr',     siren_fracerr,      'siren_fracerr/F')
    tree.Branch('ub_fracerr',        ub_fracerr,         'ub_fracerr/F')
    tree.Branch('fvis_min',          fvis_min,           'fvis_min/F')
    tree.Branch('fvis_max',          fvis_max,           'fvis_max/F')
    tree.Branch('fvis_mean',         fvis_min,           'fvis_mean/F')    

    for ibatch,batch in enumerate(dataloader):
        if ibatch%100==0:
            print(f"Batch [{ibatch}]")

        print('='*80)
        print(f"BATCH [{ibatch}]")

        with torch.no_grad():
            vox_feat, q, mask = prepare_input(batch, config, pmtpos, device)
            coord  = batch['avepos']
            Nb,Nv,Nd = coord.shape

            tstart_forward = time.time()

            print("model inputs: ")
            print("vox_feat: ",vox_feat.shape)
            #print(vox_feat)
            print("q: ",q.shape)
            print(q.reshape( (Nb,Nv,32))[0,:,0])
            print("mask.shape=",mask.shape)

            if return_fvis:
                pe_per_voxel, fvis = siren(vox_feat, q, return_fvis=return_fvis)
                print('returned fvis.shape=',fvis.shape)
                fvis = fvis.reshape( (Nb*Nv,32))
                print('fvis.reshape=',fvis.shape)
                
                #fvis_masked  = fvis.reshape(-1)[mask.reshape(-1)==1]
                fvis_masked = fvis
                print("-----------q dump (Nb,Nv,32) ---------------------------")
                print(q.reshape((Nb,Nv,32))[0,:,0])
                print("q.shape=",q.shape)
                print(q[:,0])
                print("---------- fvis_masked dump -----------------")
                print(fvis_masked)
                print("fvis_masked.shape=",fvis_masked.shape)
                print("---------- mask dump -----------------")
                print(mask)
                print("mask.shape=",mask.shape)
                print("-----------------------------------")

                fvis_mean[0] = fvis_masked.mean().item()
                fvis_max[0]  = fvis_masked.max().item()
                fvis_min[0]  = fvis_masked.min().item()
            else:
                pe_per_voxel = siren(vox_feat, q, return_fvis=return_fvis)
            print("siren model returns: ",pe_per_voxel.shape) # also per pmt
            pe_per_voxel = pe_per_voxel.reshape( (Nb,Nv,Npmt) )
            print("pe per voxel (re)shape: ",pe_per_voxel.shape) # also per pmt
            

            # mask then sum
            # reshape mask to (Nb,Nv,1) so it broadcasts to (Nb,Nv,Npmt) to match
            # pe_per_voxel
            pe_per_voxel = mask.reshape( (Nb,Nv,Npmt) )*pe_per_voxel

            # we must sum over all the relevant charge voxels per per PMT per batch entry
            # we go from
            # (B,N,P)-> (N,P)
            pe_per_pmt = pe_per_voxel.sum(dim=1)
            print('pe_per_pmt: ',pe_per_pmt.shape)

            pe_per_pmt_denorm = undo_pmt_normalization(pe_per_pmt,config)

            dt_forward = time.time()-tstart_forward
            print("forward time: ",dt_forward)

        batch_hists = []
        
        for ii in range(pe_per_pmt_denorm.shape[0]):

            siren_pe_per_pmt_t = pe_per_pmt_denorm[ii,:]
            obs_pe_per_pmt_t   = batch['observed_pe_per_pmt'][ii,:]
            ub_pe_per_pmt_t    = batch['predicted_pe_per_pmt'][ii,:]

            if debug:
                hname = f"hbatch_batch{ibatch}_{ii}"
                hists = makehists( hname, siren_pe_per_pmt_t, obs_pe_per_pmt_t, ub_pe_per_pmt_t)
                c.cd(ii+1)
                hists['obs'].Draw("hist")
                hists['siren'].Draw("histsame")
                hists['ub'].Draw("histsame")
                batch_hists.append( hists )
                c.Update()

            # calculate metrics, save to ROOT tree
            siren_petot = siren_pe_per_pmt_t.sum().item()
            obs_petot   = obs_pe_per_pmt_t.sum().item()
            ub_petot    = ub_pe_per_pmt_t.sum().item()

            siren_pdf = siren_pe_per_pmt_t/siren_petot
            obs_pdf   = (obs_pe_per_pmt_t/obs_petot).to(device)
            ub_pdf    = (ub_pe_per_pmt_t/ub_petot).to(device)

            obs_pe_tot[0]   = obs_petot
            siren_pe_tot[0] = siren_petot
            ub_pe_tot[0]    = ub_petot
            siren_sinkhorn[0] = sinkhorn_fn( siren_pdf, x_pred, obs_pdf, y_target )
            ub_sinkhorn[0]    = sinkhorn_fn( ub_pdf, x_pred, obs_pdf, y_target )
            siren_fracerr[0]  = ( siren_petot-obs_petot)/obs_petot
            ub_fracerr[0]     = ( ub_petot -obs_petot)/obs_petot

            for ipmt in range(Npmt):
                siren_pe_per_pmt_v[ipmt] = siren_pe_per_pmt_t[ipmt]
                ub_pe_per_pmt_v[ipmt]    = ub_pe_per_pmt_t[ipmt]
                obs_pe_per_pmt_v[ipmt]   = obs_pe_per_pmt_t[ipmt]

            tree.Fill()

        
        if debug:
            print("[enter] to continue.")
            input()
        
        if num_batches>0 and ibatch>=num_batches:
            break

    print('end of loop')
    tree.Write()
    fout.Close()

        


if __name__ == "__main__":

    main(sys.argv[1])
    
