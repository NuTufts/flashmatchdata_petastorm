#!/usr/bin/env python3
"""
Training script for SIREN-based light model with HDF5 FlashMatch data

This script trains a SIREN network to predict PMT light patterns from 3D voxel data
using the new FlashMatch HDF5 dataset format.
"""

import os
import sys
import time
import yaml
from math import pow, cos
from pathlib import Path
from typing import Dict, Any, Optional
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
import wandb

# Optional geomloss import for EMD loss
try:
    import geomloss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False
    print("Warning: geomloss not available - EMD loss will be disabled")

# Add data_prep to path for the new data loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_prep'))

print("TRAIN LIGHT-MODEL SIREN WITH HDF5 DATA")
tstart = time.time()

# Import FlashMatch modules
import flashmatchnet
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
from flashmatchnet.utils.pmtutils import get_2d_zy_pmtpos_tensor
from flashmatchnet.utils.trackingmetrics import validation_calculations
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings, prepare_mlp_input_variables
from flashmatchnet.utils.pmtpos import getPMTPosByOpDet, getPMTPosByOpChannel
from flashmatchnet.losses.loss_poisson_emd import PoissonNLLwithEMDLoss
from flashmatchnet.model.lightmodel_siren import LightModelSiren


# Import new HDF5 data modules
from flashmatchnet.data.read_flashmatch_hdf5 import FlashMatchVoxelDataset
from flashmatchnet.data.flashmatch_mixup import create_mixup_dataloader, MixUpFlashMatchDataset

print(f"Modules loaded: {time.time()-tstart:.2f} sec")


def parse_arguments():
    """Parse command line arguments"""
    parser = ArgumentParser(description='Train SIREN model for flash matching')
    
    parser.add_argument('--config', '-c', 
                        type=str, 
                        default='config_siren_hdf5_data.yaml',
                        help='Path to YAML configuration file')
    
    parser.add_argument('--wandb-project', 
                        type=str, 
                        default='flashmatch-siren-hdf5-data',
                        help='W&B project name')
    
    parser.add_argument('--wandb-run-name',
                        type=str,
                        default=None,
                        help='W&B run name (auto-generated if not specified)')
    
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Perform a dry run without training')
    
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file"""
    
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['train', 'logger', 'dataloader', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config file")
    
    # Set defaults for optional parameters
    defaults = {
        'train': {
            'freeze_batch': False,
            'freeze_ly_param': False,
            'mag_loss_on_sum': False,
            'start_iteration': 0,
            'num_iterations': 10000,
            'num_valid_iters': 10,
            'checkpoint_iters': 1000,
            'checkpoint_folder': './checkpoints/',
            'setup_only': False,
            'NPMTS': 32
        },
        'logger': {
            'use_wandb': True,
            'log_interval': 10,
            'save_interval': 1000
        },
        'dataloader': {
            'shuffle': True,
            'batchsize': 32,
            'num_workers': 4,
            'pin_memory': True,
            'mixup_prob': 0.5,  # Mixup probability
            'mixup_alpha': 1.0,  # Mixup beta distribution parameter
            'max_voxels': 500,   # Maximum voxels per sample
            'normalize_features': True
        },
        'model': {
            'use_cos_input_embedding_vectors': False
        }
    }
    
    # Merge defaults with loaded config
    for section, defaults_dict in defaults.items():
        if section in config:
            for key, default_value in defaults_dict.items():
                if key not in config[section]:
                    config[section][key] = default_value
    
    return config


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """Create training and validation data loaders using the new HDF5 dataset"""
    
    dataloader_config = config['dataloader']
    
    # Create base datasets
    print(f"Loading training data from: {dataloader_config['train_filelist']}")
    train_base_dataset = FlashMatchVoxelDataset(
        hdf5_files=dataloader_config['train_filelist'],
        max_voxels=dataloader_config.get('max_voxels', 500),
        load_to_memory=False
    )
    
    print(f"Loading validation data from: {dataloader_config['valid_filelist']}")
    valid_base_dataset = FlashMatchVoxelDataset(
        hdf5_files=dataloader_config['valid_filelist'],
        max_voxels=dataloader_config.get('max_voxels', 500),
        load_to_memory=False
    )
    
    # Apply MixUp augmentation if specified
    mixup_prob = dataloader_config.get('mixup_prob')
    
    if mixup_prob > 0:
        print(f"Applying MixUp augmentation with probability {mixup_prob}")
        
        # Create MixUp datasets
        train_dataset = MixUpFlashMatchDataset(
            base_dataset=train_base_dataset,
            mixup_prob=mixup_prob,
            alpha=dataloader_config.get('mixup_alpha', 1.0),
            max_total_voxels=dataloader_config.get('max_voxels', 500) * 2
        )
        
        # No MixUp for validation
        valid_dataset = valid_base_dataset
        
        # Use custom collate function for MixUp
        from flashmatchnet.data.flashmatch_mixup import mixup_collate_fn
        collate_fn = mixup_collate_fn
    else:
        train_dataset = train_base_dataset
        valid_dataset = valid_base_dataset
        collate_fn = None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dataloader_config['batchsize'],
        shuffle=dataloader_config['shuffle'],
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config.get('pin_memory', True),
        collate_fn=collate_fn if mixup_prob > 0 else None
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=dataloader_config['batchsize'],
        shuffle=False,  # Don't shuffle validation
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config.get('pin_memory', True)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Batch size: {dataloader_config['batchsize']}")
    print(f"Training batches per epoch: {len(train_dataloader)}")
    
    return train_dataloader, valid_dataloader, len(train_dataset), len(valid_dataset)


def create_models(config: Dict[str, Any], device: torch.device) -> tuple:
    """Create the FlashMatchMLP and SIREN models"""
    
    model_config = config['model']
    
    # Create MLP for embeddings
    flashmlp_config = model_config['flashmlp']
    mlp = FlashMatchMLP(
        input_nfeatures=flashmlp_config['input_nfeatures'],
        hidden_layer_nfeatures=flashmlp_config['hidden_layer_nfeatures']
    ).to(device)
    
    # Create SIREN network
    siren_config = model_config['lightmodelsiren']
    
    # Handle final activation
    if siren_config.get('final_activation') == 'identity':
        final_activation = nn.Identity()
    else:
        raise ValueError(f"Invalid final_activation: {siren_config.get('final_activation')}")
    
    # Create SIREN model
    siren = LightModelSiren(
        dim_in=siren_config['dim_in'],
        dim_hidden=siren_config['dim_hidden'],
        dim_out=siren_config['dim_out'],
        num_layers=siren_config['num_layers'],
        w0_initial=siren_config['w0_initial'],
        final_activation=final_activation,
        use_logpe=config.get('use_logpe')
    ).to(device)
    
    return mlp, siren

def create_loss_functions(device):
    loss_fn_train = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                          mag_loss_on_sum=False,
                                          full_poisson_calc=False).to(device)
    loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                          mag_loss_on_sum=False,
                                          full_poisson_calc=True).to(device)
    return loss_fn_train, loss_fn_valid


def setup_wandb(config: Dict[str, Any], model, args) -> Optional[Any]:
    """Initialize Weights & Biases logging"""
    
    logger_config = config['logger']
    
    if not logger_config.get('use_wandb', True):
        return None
    
    print("Setting up Weights & Biases logging...")
    
    # Initialize W&B
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config,
        reinit=True
    )
    
    # Log the configuration
    wandb.config.update(config)
    
    # Add additional metadata
    wandb.config.update({
        'hostname': os.uname().nodename,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'pytorch_version': torch.__version__,
    })

    wandb.watch(model, log="all", log_freq=1000) 
    
    return run

def create_pmtpos():
    # copy position data into numpy array format
    pmtpos = torch.zeros( (32, 3) )
    for i in range(32):
        opdetpos = getPMTPosByOpDet(i,use_v4_geom=True)
        for j in range(3):
            pmtpos[i,j] = opdetpos[j]
    # change coordinate system to 'tensor' system
    # main difference is y=0 is at bottom of TPC        
    pmtpos[:,1] -= -117.0
    # The pmt x-positions are wrong (!).
    # They would be in the TPC with the values I have stored.
    # So move them outside the TPC
    pmtpos[:,0] = -20.0
    # now corrected to be at -11, but need to keep things consistent
    return pmtpos

def save_checkpoint(models: Dict[str, nn.Module], 
                   optimizers: Dict[str, Any],
                   iteration: int,
                   config: Dict[str, Any],
                   checkpoint_path: str):
    """Save training checkpoint"""
    
    checkpoint = {
        'iteration': iteration,
        'config': config,
        'model_states': {},
        'optimizer_states': {},
    }
    
    # Save model states
    for name, model in models.items():
        checkpoint['model_states'][name] = model.state_dict()
    
    # Save optimizer states
    for name, optimizer in optimizers.items():
        if optimizer is not None:
            checkpoint['optimizer_states'][name] = optimizer.state_dict()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str,
                   models: Dict[str, nn.Module],
                   optimizers: Dict[str, Any] = None) -> int:
    """Load training checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model states
    for name, model in models.items():
        if name in checkpoint['model_states']:
            model.load_state_dict(checkpoint['model_states'][name])
    
    # Load optimizer states if provided
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            if optimizer is not None and name in checkpoint['optimizer_states']:
                optimizer.load_state_dict(checkpoint['optimizer_states'][name])
    
    iteration = checkpoint.get('iteration', 0)
    print(f"Resumed from iteration {iteration}")
    
    return iteration


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

def get_learning_rate( epoch, warmup_epoch, cosine_epoch_period, warmup_lr, cosine_max_lr, cosine_min_lr ):
    if epoch < warmup_epoch:
        return warmup_lr
    elif epoch>=warmup_epoch and epoch-warmup_epoch<cosine_epoch_period:
        lr = cosine_min_lr + 0.5*(cosine_max_lr-cosine_min_lr)*(1+cos( (epoch-warmup_epoch)/float(cosine_epoch_period)*3.14159 ) )
        return lr
    else:
        return cosine_min_lr

def run_validation( config, iteration, epoch, net, loss_fn_valid, valid_iter, valid_dataloader, device, pmtpos ):

    with torch.no_grad():
        print("=========================")
        print("[Validation at iter=",iteration,"]")            
        print("Epoch: ",epoch)
        print("LY=",net.get_light_yield().cpu().item())
        #print("(next) learning_rate=",next_lr," :  lr_LY=",next_lr*0.01)
        print("=========================")
        
        print("Run validation calculations on valid data") 
        valid_metrics = {
            "loss_tot_ave":0.0,
            "loss_emd_ave":0.0,
            "loss_mag_ave":0.0
        }

        for ival in range(config['train'].get('num_valid_batches')):
            try:
                valid_batch = next(valid_iter)
            except:
                print('reset validation data iterator')
                valid_iter = iter(valid_dataloader)
            apply_normalization(valid_batch,config)
            valid_iter_dict = validation_calculations( valid_batch, net, 
                    loss_fn_valid, config['dataloader'].get('batchsize'),
                    device, pmtpos,
                    nvalid_iters=config['train'].get('num_valid_batches'),
                    use_embed_inputs=config['model'].get('use_cos_input_embedding_vectors') )
            for k in valid_metrics:
                valid_metrics[k] += valid_iter_dict[k]
        
        print("  [valid total loss]: ",valid_metrics["loss_tot_ave"])
        print("  [valid emd loss]:   ",valid_metrics["loss_emd_ave"])
        print("  [valid mag loss]:   ",valid_metrics["loss_mag_ave"])
        print("Run validation calculations on train data")
        # train_info_dict = validation_calculations( train_iter, net, loss_fn_valid, BATCHSIZE, device, NVALID_ITERS,
        #                                            mlp=mlp,
        #                                            use_embed_inputs=USE_COS_INPUT_EMBEDDING_VECTORS)
        # print("  [train total loss]: ",train_info_dict["loss_tot_ave"])
        # print("  [train emd loss]: ",train_info_dict["loss_emd_ave"])
        # print("  [train mag loss]: ",train_info_dict["loss_mag_ave"])  

        return valid_metrics          


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, valid_loader, num_train, num_valid = create_data_loaders(config)
    
    # Calculate iterations
    train_config = config['train']
    train_iters_per_epoch = len(train_loader)
    
    if train_config.get('freeze_batch', False):
        # For debugging: use single batch
        num_train = config['dataloader']['batchsize']
        num_valid = config['dataloader']['batchsize']
        train_iters_per_epoch = 1

    # Load PMT positions
    pmtpos = create_pmtpos().to(device)
    
    # Create models
    model_config = config['model']
    mlp, siren = create_models(config, device)
    
    # Print model architectures
    # print("\nFlashMatchMLP architecture:")
    # print(mlp)
    # print(f"Total parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    print("\nSIREN architecture:")
    print(siren)
    print(f"Total parameters: {sum(p.numel() for p in siren.parameters()):,}")
    
    # Setup optimizers
    lr_config = train_config.get('learning_rate_config', {})
    learning_rate = lr_config.get('max', 1e-3)
    
    param_group_main = []
    param_group_ly   = []

    for name,param in siren.named_parameters():
        if name=="light_yield":
            if train_config['freeze_ly_param']:
                param.requires_grad = False
            param_group_ly.append(param)
        else:
            param_group_main.append(param)
    param_group_list = [{"params":param_group_ly,"lr":lr_config['warmup_lr']*0.1,"weight_decay":1.0e-5},
                        {"params":param_group_main,"lr":lr_config['warmup_lr']}]
    optimizer = torch.optim.AdamW( param_group_list )

    # Setup loss function
    loss_fn_train, loss_fn_valid = create_loss_functions(device)
    
    # Load checkpoint if resuming
    start_iteration = train_config.get('start_iteration', 0)
    if args.resume:
        start_iteration = load_checkpoint(
            args.resume, 
            {'mlp':mlp,'siren': siren},
            {'mlp':mlp,'siren': optimizer}
        )
    
    # Setup W&B logging
    wandb_run = setup_wandb(config, siren, args) if not args.dry_run else None
    
    # Log configuration summary
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_valid}")
    print(f"Batch size: {config['dataloader']['batchsize']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Start iteration: {start_iteration}")
    print(f"Total iterations: {train_config['num_iterations']}")
    print(f"Checkpoint interval: {train_config.get('checkpoint_iters', 1000)}")
    print(f"Validation interval: {train_config.get('num_valid_iters', 10)}")
    print(f"MixUp probability: {config['dataloader'].get('mixup_prob', 0.5)}")
    print(f"MixUp alpha: {config['dataloader'].get('mixup_alpha')}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    if args.dry_run:
        print("DRY RUN COMPLETE - Exiting without training")
        return
    
    # Training loop
    print("Training loop setup complete!")
    print("TODO: Implement full training loop with:")
    print("  - Forward pass through models")
    print("  - Loss calculation")
    print("  - Backward pass and optimization")
    print("  - Validation loop")
    print("  - Checkpoint saving")
    print("  - W&B logging")

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    epoch = 0.0
    last_iepoch = -1
    end_iteration = start_iteration + train_config['num_iterations']

    for iteration in range(start_iteration,end_iteration): 

        siren.train()
        mlp.train()
        optimizer.zero_grad()
    
        tstart_dataprep = time.time()

        if train_config['freeze_batch']:
            if iteration==0:
                batch = next(train_iter)
            else:
                pass # intentionally not changing data
        else:
            batch = next(train_iter)

        if train_config['freeze_batch']:
            epoch += 1.0
        else:
            epoch = float(iteration)/float(train_iters_per_epoch)


        #print(batch.keys())
        with torch.no_grad():
            apply_normalization(batch,config)

        # coord = row['coord'].to(device)
        coord  = batch['avepos'].to(device)

        Nb,Nv,Nd = coord.shape
        #print("Coordinate shape: Nb, Nv, Nd: ",Nb," ",Nv," ",Nd)

        q_feat = batch['planecharge_normalized'].to(device)
        mask   = batch['mask'].to(device)
        n_voxels = batch['n_voxels'].to(device)
        start_per_batch = torch.zeros( config["dataloader"].get('batchsize'), dtype=torch.int64 ).to(device)
        end_per_batch   = torch.zeros( config["dataloader"].get('batchsize'), dtype=torch.int64 ).to(device)
        for i in range(config['dataloader'].get('batchsize')):
            start_per_batch[i] = i*Nv
            end_per_batch[i]   = (i+1)*Nv

        # print('n_voxels: ',n_voxels)
        # print('coord: ',coord.shape)
        # print('q_feat: ',q_feat.shape)
        # print('start_per_batch: ',start_per_batch.shape)
        # print('mask: ',mask.shape)
        # print('num_ones: ',(mask==1).sum())
        # print('num_zeros: ',(mask==0).sum())
        # print('start_per_batch: ',start_per_batch)
        

        # for each coord, we produce the other features
        with torch.no_grad():
            if model_config.get('use_cos_input_embedding_vectors'):
                vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, pmtpos, vox_len_cm=1.0 )
            else:
                # takes in (N,3) coord and (N,3) charge tensor
                # before coord and q_feat go in,
                # they get reshaped from (Nb,Nv,3) -> (Nb*Nv,3)
                vox_feat = prepare_mlp_input_variables( coord.reshape(-1,3), q_feat.reshape(-1,3), pmtpos, vox_len_cm=1.0 )
                #print('vox feat: ',vox_feat.shape,' device=',vox_feat.device)
        
        dt_dataprep = time.time()-tstart_dataprep
        print(f"iter[{iteration}] dt_dataprep={dt_dataprep:0.2f} secs")

        tstart_forward = time.time()
        Nbv,Npmt,K = vox_feat.shape
        vox_feat = vox_feat.reshape( (Nbv*Npmt,K) )
        q = vox_feat[:,-1:]
        vox_feat = vox_feat[:,:-1]
        K += -1

        with torch.no_grad():
            print("INPUTS ==================")
            print("vox_feat.shape=",vox_feat.shape," from (Nb x Nv,Npmt,Nk)=",(Nbv,Npmt,K))
            print("q.shape=",q.shape)

        pe_per_voxel = siren(vox_feat, q)
        print("siren model returns: ",pe_per_voxel.shape) # also per pmt
        pe_per_voxel = pe_per_voxel.reshape( (Nb,Nv,Npmt) )

        # mask then sum
        # reshape mask to (Nb,Nv,1) so it broadcasts to (Nb,Nv,Npmt) to match
        # pe_per_voxel
        pe_per_voxel = mask.reshape( (Nb,Nv,1))*pe_per_voxel

        # we must sum over all the relevant charge voxels per per PMT per batch entry
        # we go from
        # (B,N,P)-> (N,P)
        pe_per_voxel = pe_per_voxel.sum(dim=1)

        dt_forward = time.time()-tstart_forward
        print("forward time: ",dt_forward)

        if False:
            # for debug
            print("PE_PER_VOXEL ================")
            print("output prediction: pe_per_voxel.shape=",pe_per_voxel.shape)
            print("output prediction: pe_per_voxel stats: ")
            with torch.no_grad():
                print(" mean: ",pe_per_voxel.mean())
                print(" var: ",pe_per_voxel.var())
                print(" max: ",pe_per_voxel.max())
                print(" min: ",pe_per_voxel.min())
                print(pe_per_voxel)
                #print(pe_per_voxel[rand_entries[:],:2])

        # get target pe per pmt
        pe_per_pmt_target = batch['observed_pe_per_pmt_normalized'].to(device)
        with torch.no_grad():
            target_pe_sum = pe_per_pmt_target.sum(dim=1)
            # print('Target pe_per_pmt: ',pe_per_pmt_target.shape)
            # print('Target pe sum: ',target_pe_sum)
            # print('Target pe: ',pe_per_pmt_target)

        loss_tot,(floss_tot,floss_emd,floss_mag,pred_pesum,pred_pemax) = \
            loss_fn_train( pe_per_voxel, pe_per_pmt_target, 
                            start_per_batch, end_per_batch, mask=mask )

        if True:
            # for debug
            with torch.no_grad():
                print("loss: ")
                print("  total: ",floss_tot)
                print("    emd: ",floss_emd)
                print("    mag: ",floss_mag)
                #print("  predicted pe sum: ",pred_pesum)
                #print("  predicted pe max: ",pred_pemax)

        # ----------------------------------------
        # Backprop
        dt_backward = time.time()

        loss_tot.backward()
        nn.utils.clip_grad_norm_(siren.parameters(), 1.0)
        optimizer.step()

        # get the lr we just used
        iter_lr    = optimizer.param_groups[1]["lr"]
        iter_lr_ly = optimizer.param_groups[0]["lr"]
        next_lr = get_learning_rate( epoch, 
                                    lr_config['warmup_nepochs'], 
                                    lr_config['cosine_nepochs'],
                                    lr_config['warmup_lr'],
                                    lr_config['max'], lr_config['min'] )
        optimizer.param_groups[1]["lr"] = next_lr
        optimizer.param_groups[0]["lr"] = next_lr*0.01 # LY learning rate

        dt_backward = time.time()-dt_backward

        print("backward time: ",dt_backward," secs")

        if iteration>start_iteration and iteration%int(config['train'].get('checkpoint_iters'))==0:
            print('save checkpoint')
            save_checkpoint({'siren':siren},
                {'siren':optimizer},
                iteration,
                config,
                config['train'].get('checkpoint_folder')+"/checkpoint_iteration_%08d.pt"%(iteration) )

        if iteration>0 and iteration%int(config['train'].get('num_valid_iters'))==0:
            with torch.no_grad():
                valid_info_dict = run_validation( config, iteration, epoch, siren, 
                                                    loss_fn_valid, valid_iter, valid_loader, device, pmtpos )

                train_info_dict = run_validation( config, iteration, epoch, siren, 
                                                    loss_fn_train, train_iter, train_loader, device, pmtpos )
        
                if config['logger'].get('use_wandb'):
                    for_wandb = {
                        "loss_tot_ave_train":train_info_dict["loss_tot_ave"],
                        "loss_emd_ave_train":train_info_dict["loss_emd_ave"],
                        "loss_mag_ave_train":train_info_dict["loss_mag_ave"],
                        "loss_tot_ave_valid":valid_info_dict["loss_tot_ave"],
                        "loss_emd_ave_valid":valid_info_dict["loss_emd_ave"],
                        "loss_mag_ave_valid":valid_info_dict["loss_mag_ave"],
                        #"pesum_v_x_pred":wandb.plot.scatter(tabledata, "x", "pe_sum_pred",
                        #                                    title="PE sum vs. x (cm)"),
                        #"pesum_v_q_pred":wandb.plot.scatter(tabledata, "qmean","pe_sum_pred",
                        #                                    title="PE sum vs. q sum"),
                        "lr":iter_lr,
                        "lr_lightyield":iter_lr_ly,
                        "lightyield":siren.get_light_yield().cpu().item(),
                        "epoch":epoch}
                    wandb.log(for_wandb, step=iteration)

        
        if False:
            # for debug
            break

    print("FINISHED ITERATION LOOP!")
    save_checkpoint({'siren':siren},
        {'siren':optimizer},
        iteration,
        config,
        config['train'].get('checkpoint_folder')+"/checkpoint_iteration_%08d.pt"%(end_iteration))
    

    


    # Close W&B run
    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
