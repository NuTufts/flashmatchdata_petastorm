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
import geomloss

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
from data_prep.read_flashmatch_hdf5 import FlashMatchVoxelDataset
from data_prep.flashmatch_mixup import create_mixup_dataloader, MixUpFlashMatchDataset

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
        from data_prep.flashmatch_mixup import mixup_collate_fn
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
        final_activation=final_activation
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


def setup_wandb(config: Dict[str, Any], args) -> Optional[Any]:
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
            log_pe = torch.log(1.0 + batch['observed_pe_per_pmt'])
            batch['observed_pe_per_pmt_normalized'] = (log_pe - offset) / scale
    
    # Normalize plane charge if specified
    if 'planecharge_norm_params' in norm_config:
        charge_params = norm_config['planecharge_norm_params']
        offsets = torch.tensor(charge_params['offset'], device=batch['planecharge'].device)
        scales = torch.tensor(charge_params['scale'], device=batch['planecharge'].device)
        
        # Apply log transform and normalization to plane charge
        if 'planecharge' in batch:
            log_charge = torch.log(1.0 + batch['planecharge'])
            batch['planecharge_normalized'] = (log_charge + offsets) / scales
    
    return batch

def get_learning_rate( epoch, warmup_epoch, cosine_epoch_period, warmup_lr, cosine_max_lr, cosine_min_lr ):
    if epoch < warmup_epoch:
        return warmup_lr
    elif epoch>=warmup_epoch and epoch-warmup_epoch<cosine_epoch_period:
        lr = cosine_min_lr + 0.5*(cosine_max_lr-cosine_min_lr)*(1+cos( (epoch-warmup_epoch)/float(cosine_epoch_period)*3.14159 ) )
        return lr
    else:
        return cosine_min_lr

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
    print("\nFlashMatchMLP architecture:")
    print(mlp)
    print(f"Total parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    print("\nSIREN architecture:")
    print(siren)
    print(f"Total parameters: {sum(p.numel() for p in siren.parameters()):,}")
    
    # Setup optimizers
    lr_config = train_config.get('learning_rate_config', {})
    learning_rate = lr_config.get('max', 1e-3)
    
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
    optimizer_siren = torch.optim.Adam(siren.parameters(), lr=learning_rate)
    
    # Setup loss function
    loss_fn_train, loss_fn_valid = create_loss_functions(device)
    
    # Load checkpoint if resuming
    start_iteration = train_config.get('start_iteration', 0)
    if args.resume:
        start_iteration = load_checkpoint(
            args.resume, 
            {'mlp': mlp, 'siren': siren},
            {'mlp': optimizer_mlp, 'siren': optimizer_siren}
        )
    
    # Setup W&B logging
    wandb_run = setup_wandb(config, args) if not args.dry_run else None
    
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


        print(batch.keys())
        apply_normalization(batch,config['dataloader'])

        # coord = row['coord'].to(device)
        coord  = batch['use_cos_input_embedding_vectors'].to(device)
        q_feat = batch['planecharge_normalized'].to(device)

        # for each coord, we produce the other features
        with torch.no_grad():
            if model_config.use_cos_input_embedding_vectors:
                vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, pmtpos, vox_len_cm=1.0 )
            else:
                vox_feat, q = prepare_mlp_input_variables( coord, q_feat, pmtpos, vox_len_cm=1.0 )


        print('vox feat: ',vox_feat.shape)
        print('q: ',q.shape)
        # entries_per_batch = row['batchentries']
        # start_per_batch = torch.from_numpy(row['batchstart']).to(device)
        # end_per_batch   = torch.from_numpy(row['batchend']).to(device)

        # # for each coord, we produce the other features
        # with torch.no_grad():
        #     if USE_COS_INPUT_EMBEDDING_VECTORS:
        #         vox_feat, q = prepare_mlp_input_embeddings( coord, q_feat, mlp )
        #     else:
        #         vox_feat, q = prepare_mlp_input_variables( coord, q_feat, mlp )
        dt_dataprep = time.time()-tstart_dataprep
        print(f"iter[{iteration}] dt_dataprep={dt_dataprep:0.2f} secs")

        if True:
            break
    
    # Close W&B run
    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()