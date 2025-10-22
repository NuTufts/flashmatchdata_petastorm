#!/usr/bin/env python3
"""
Training script for SIREN-based light model with HDF5 FlashMatch data

This script trains a SIREN network to predict PMT light patterns from 3D voxel data
using the new FlashMatch HDF5 dataset format.
"""

import os
import sys

sys.path.append( os.environ['GEOMLOSS_DIR'] )
sys.path.append( os.environ['SIREN_DIR'] )

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
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
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
from flashmatchnet.losses.loss_unbalanced_sinkhorn import UnbalancedSinkhornLoss
from flashmatchnet.model.lightmodel_siren import LightModelSiren
from flashmatchnet.utils.multigpu import setup_distributed, cleanup_distributed


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

    # Distributed training arguments (usually set by torchrun)
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    
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
            'distribution':'uniform',
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


def create_data_loaders(config: Dict[str, Any], is_distributed: bool, rank: int, world_size: int) -> tuple:
    """Create training and validation data loaders using the new HDF5 dataset"""
    
    dataloader_config = config['dataloader']
    
    # Create base datasets
    if rank == 0:
        print(f"Loading training data from: {dataloader_config['train_filelist']}")
    train_base_dataset = FlashMatchVoxelDataset(
        hdf5_files=dataloader_config['train_filelist'],
        max_voxels=dataloader_config.get('max_voxels', 500),
        load_to_memory=False
    )
    
    if rank == 0:
        print(f"Loading validation data from: {dataloader_config['valid_filelist']}")
    valid_base_dataset = FlashMatchVoxelDataset(
        hdf5_files=dataloader_config['valid_filelist'],
        max_voxels=dataloader_config.get('max_voxels', 500),
        load_to_memory=False
    )
    
    # Apply MixUp augmentation if specified
    mixup_prob = dataloader_config.get('mixup_prob')
    
    if mixup_prob > 0:
        if rank==0:
            print(f"Applying MixUp augmentation with probability {mixup_prob}")
        
        # Create MixUp datasets
        train_dataset = MixUpFlashMatchDataset(
            base_dataset=train_base_dataset,
            distribution=dataloader_config.get('distribution','uniform'),
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
    
    # Create samplers for distributed training
    train_sampler = None
    valid_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=dataloader_config['shuffle']
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle_train = False  # Sampler handles shuffling
    else:
        shuffle_train = dataloader_config['shuffle']

    # Adjust batch size for distributed training
    # Each GPU gets batchsize/world_size samples
    effective_batch_size = dataloader_config['batchsize']
    if is_distributed:
        if config['distributed'].get('scale_batch_size', True):
            # Keep per-GPU batch size constant, total batch size scales with GPUs
            pass
        else:
            # Keep total batch size constant, divide among GPUs
            effective_batch_size = max(1, dataloader_config['batchsize'] // world_size)


    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle_train if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config.get('pin_memory', True),
        collate_fn=collate_fn if mixup_prob > 0 else None,
        drop_last=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle_train if train_sampler is None else False,
        sampler=valid_sampler,
        num_workers=dataloader_config['num_workers'],
        pin_memory=dataloader_config.get('pin_memory', True),
        drop_last=True
    )
    
    if rank==0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}")
        print(f"Effective batch size per GPU: {effective_batch_size}")
        print(f"Total batch size: {effective_batch_size * world_size}")
        print(f"Training batches per epoch: {len(train_dataloader)}")
    
    return train_dataloader, valid_dataloader, len(train_dataset), len(valid_dataset), train_sampler, valid_sampler


def create_models(config: Dict[str, Any], device: torch.device, is_distributed: bool, local_rank: int) -> tuple:
    """Create the FlashMatchMLP and SIREN models with DDP wrapper if distributed"""

    model_config = config['model']
    network_type = model_config.get('network_type')
    
    if network_type=='mlp':
        # Create MLP for embeddings
        flashmlp_config = model_config['mlp']
        model = FlashMatchMLP(
            input_nfeatures=flashmlp_config['input_nfeatures'],
            hidden_layer_nfeatures=flashmlp_config['hidden_layer_nfeatures'],
            norm_layer=flashmlp_config['norm_layer']
        ).to(device)
    elif network_type=='lightmodelsiren':

        # Create SIREN network
        siren_config = model_config['lightmodelsiren']

        # Handle final activation
        if siren_config.get('final_activation') == 'identity':
            final_activation = nn.Identity()
        else:
            raise ValueError(f"Invalid final_activation: {siren_config.get('final_activation')}")
    
        # Create SIREN model
        model = LightModelSiren(
            dim_in=siren_config['dim_in'],
            dim_hidden=siren_config['dim_hidden'],
            dim_out=siren_config['dim_out'],
            num_layers=siren_config['num_layers'],
            w0_initial=siren_config['w0_initial'],
            final_activation=final_activation
        ).to(device)
    else:
        raise ValueError(f"Invalid network_type given: {network_type}")

    # Wrap models with DDP if distributed
    if is_distributed:
        # Sync batch norm layers across GPUs if present
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=config['distributed'].get('gradient_as_bucket_view', True),
            find_unused_parameters=config['distributed'].get('find_unused_parameters', False),
            broadcast_buffers=config['distributed'].get('broadcast_buffers', True)
        )
    
    return model

def create_loss_functions(device, train_config, batchsize):

    loss_config = train_config.get("loss",{})
    if "name" not in loss_config:
        raise ValueError("Loss configuation block needs a 'name' parameter to choose loss type")

    loss_name = loss_config.get("name")
    if loss_name=="poissonemd":
        loss_fn_train = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                              mag_loss_on_sum=False,
                                              full_poisson_calc=False).to(device)
        loss_fn_valid = PoissonNLLwithEMDLoss(magloss_weight=1.0,
                                              mag_loss_on_sum=False,
                                              full_poisson_calc=True).to(device)
    elif loss_name=="unbalanced_sinkhorn":
        loss_fn_train = UnbalancedSinkhornLoss(batchsize)
        loss_fn_valid = UnbalancedSinkhornLoss(batchsize)
    else:
        raise ValueError("Unrecognized loss function name: ",loss_name)
        
    return loss_fn_train, loss_fn_valid


def setup_wandb(config: Dict[str, Any], model, args, rank: int) -> Optional[Any]:
    """Initialize Weights & Biases logging"""

    if rank != 0:
        return None
    
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
        'distributed_enabled': dist.is_initialized(),
        'world_size': dist.get_world_size() if dist.is_initialized() else 1,
    })

    wandb.watch(model, log="all", log_freq=1000) 
    
    return run

def create_pmtpos(apply_y_offset=False):
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

def save_checkpoint(models: Dict[str, nn.Module], 
                   optimizers: Dict[str, Any],
                   iteration: int,
                   config: Dict[str, Any],
                   checkpoint_path: str,
                   rank: int):
    """Save training checkpoint (only on rank 0)"""

    if rank != 0:
        return
    
    checkpoint = {
        'iteration': iteration,
        'config': config,
        'model_states': {},
        'optimizer_states': {},
    }
    
    # Save model states
    for name, model in models.items():
        if isinstance(model,DDP):
            checkpoint['model_states'][name] = model.module.state_dict()
        else:
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
                   rank: int, 
                   optimizers: Dict[str, Any] = None) -> int:
    """Load training checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if rank != 0:
        return

    print(f"Loading checkpoint from {checkpoint_path}")

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model states
    for name, model in models.items():
        if name in checkpoint['model_states']:
            if isinstance(model,DDP):
                model.module.load_state_dict(checkpoint['model_states'][name])
            else:
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

def run_validation( config, iteration, epoch, net, loss_fn_valid, valid_iter, valid_dataloader,
                   valid_sampler, device, pmtpos, rank, is_train_eval=False ):
    """
    Run validation on dataset.

    Args:
        is_train_eval: If True, this is evaluating training data (not actual validation)

    Returns:
        valid_metrics: Dictionary of validation metrics
        valid_iter: Updated iterator (important for maintaining state)
    """

    with torch.no_grad():
        if rank==0:
            print("=========================")
            print("[Validation at iter=",iteration,"]")
            print("Epoch: ",epoch)
            if isinstance(net,DDP):
                print("LY=",net.module.get_light_yield().cpu().item())
            else:
                print("LY=",net.get_light_yield().cpu().item())
            print("=========================")
            if is_train_eval:
                print("Run validation calculations on train data")
            else:
                print("Run validation calculations on valid data")

        valid_metrics = {
            "loss_tot_ave":0.0,
            "loss_emd_ave":0.0,
            "loss_mag_ave":0.0
        }

        effective_batch_size = config['dataloader'].get('batchsize')
        if dist.is_initialized():
            world_size = dist.get_world_size()
            if not config['distributed'].get('scale_batch_size'):
                effective_batch_size = max(1, effective_batch_size // world_size)

        num_valid_batches_collected = 0
        num_valid_batches_needed = config['train'].get('num_valid_batches')
        validation_epoch = 0

        # Proper iteration through validation data
        while num_valid_batches_collected < num_valid_batches_needed:
            try:
                valid_batch = next(valid_iter)
            except StopIteration:
                # End of dataset reached - create new iterator with new epoch
                validation_epoch += 1
                if rank==0:
                    print(f'Validation dataset exhausted, creating new iterator (epoch {validation_epoch})')

                # Update sampler epoch for proper distributed shuffling
                if valid_sampler is not None:
                    valid_sampler.set_epoch(int(epoch) * 1000 + validation_epoch)  # Use unique epoch number

                valid_iter = iter(valid_dataloader)
                valid_batch = next(valid_iter)

            # check batch size - skip incomplete batches rather than restarting
            Nb, Nv, K = valid_batch['avepos'].shape
            if Nb != effective_batch_size:
                if rank==0:
                    print(f"Skipping incomplete batch (size {Nb}, expected {effective_batch_size})")
                continue

            apply_normalization(valid_batch,config)
            valid_iter_dict = validation_calculations(
                valid_batch, net,
                loss_fn_valid, effective_batch_size,
                device, pmtpos,
                nvalid_iters=num_valid_batches_needed,
                use_embed_inputs=config['model'].get('use_cos_input_embedding_vectors')
            )
            for k in valid_metrics:
                valid_metrics[k] += valid_iter_dict[k]

            num_valid_batches_collected += 1

        # Reduce metrics across all GPUs if distributed
        if dist.is_initialized():
            for k in valid_metrics:
                tensor = torch.tensor(valid_metrics[k], device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                valid_metrics[k] = tensor.item()/dist.get_world_size()

        if rank==0:
            print("  [valid total loss]: ",valid_metrics["loss_tot_ave"])
            print("  [valid emd loss]:   ",valid_metrics["loss_emd_ave"])
            print("  [valid mag loss]:   ",valid_metrics["loss_mag_ave"])

        return valid_metrics, valid_iter          


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()

    # Setup distributed training
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # Load configuration
    if rank==0:
        print(f"Loading configuration from: {args.config}")
        print(f"Running in distributed mode: {is_distributed}")

    config = load_config(args.config)
    debug = config['train'].get('debug',False)    
    
    # Set device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    
    if rank==0:
        print(f"Using device: {device}")
        if is_distributed:
            print(f"World size: {world_size}")

    # Set random seeds for reproducibility
    if config['system'].get('seed') is not None:
        torch.manual_seed(config['system']['seed'] + rank)
        np.random.seed(config['system']['seed'] + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config['system']['seed'] + rank)

    
    # Create data loaders
    train_loader, valid_loader, num_train, num_valid, train_sampler, valid_sampler = create_data_loaders(
        config, is_distributed, rank, world_size
    )
    
    # Calculate iterations
    train_config = config['train']
    train_iters_per_epoch = len(train_loader)
    valid_iters_per_epoch = len(valid_loader)
    effective_batch_size  = config['dataloader']['batchsize']
    if is_distributed:
        if config['distributed'].get('scale_batch_size', True):
            # Keep per-GPU batch size constant, total batch size scales with GPUs
            pass
        else:
            # Keep total batch size constant, divide among GPUs
            effective_batch_size = max(1, effective_batch_size // world_size)
    
    
    if train_config.get('freeze_batch', False):
        # For debugging: use single batch
        num_train = config['dataloader']['batchsize']
        num_valid = config['dataloader']['batchsize']
        train_iters_per_epoch = 1

    # Load PMT positions
    pmtpos = create_pmtpos().to(device)
    
    # Create models
    model_config = config['model']
    siren = create_models(config, device, is_distributed, local_rank)
    
    # Print model architectures
    # print("\nFlashMatchMLP architecture:")
    # print(mlp)
    # print(f"Total parameters: {sum(p.numel() for p in mlp.parameters()):,}")
    
    if rank==0:
        print("\nSIREN architecture:")
        if isinstance(siren,DDP):
            print(siren.module)
            print(f"Total parameters: {sum(p.numel() for p in siren.module.parameters()):,}")
        else:
            print(siren)
            print(f"Total parameters: {sum(p.numel() for p in siren.parameters()):,}")


    # Setup optimizers
    lr_config = train_config.get('learning_rate_config', {})
    learning_rate = lr_config.get('max', 1e-3)

    # Handle DDP wrapped models for parameter groups
    siren_for_params = siren.module if isinstance(siren, DDP) else siren

    param_group_main = []
    param_group_ly   = []

    for name,param in siren.named_parameters():
        if name=="light_yield":
            if train_config['freeze_ly_param']:
                param.requires_grad = False
            param_group_ly.append(param)
        else:
            param_group_main.append(param)

    init_lr_lightield = lr_config['warmup_lr']*1.0e-5
    weight_decay = train_config['weight_decay']
        
    param_group_list = [
        {"params":param_group_ly,  "lr":init_lr_lightield,      "weight_decay":weight_decay},
        {"params":param_group_main,"lr":lr_config['warmup_lr'], "weight_decay":weight_decay},
    ]            
    optimizer = torch.optim.AdamW( param_group_list )

    # Setup loss function
    loss_fn_train, loss_fn_valid = create_loss_functions(device, config['train'], effective_batch_size)
    
    # Load checkpoint if resuming
    start_iteration = train_config.get('start_iteration', 0)
    if args.resume:
        load_checkpoint(
            args.resume, 
            rank,
            {'siren': siren},
            {'siren': optimizer}
        )
    elif train_config.get('load_from_checkpoint')==True:
        load_checkpoint(
            train_config.get('checkpoint_file'),
            {'siren': siren},
            rank,
            {'siren': optimizer}
        )
    
    # Setup W&B logging
    wandb_run = setup_wandb(config, siren, args, rank) if not args.dry_run else None
    
    # Log configuration summary
    if rank == 0:
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_valid}")
        print(f"Batch size per GPU: {config['dataloader']['batchsize'] // world_size if not config['distributed'].get('scale_batch_size', False) else config['dataloader']['batchsize']}")
        print(f"Total batch size: {config['dataloader']['batchsize'] if not config['distributed'].get('scale_batch_size', False) else config['dataloader']['batchsize'] * world_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Start iteration: {start_iteration}")
        print(f"Total iterations: {train_config['num_iterations']}")
        print(f"Checkpoint interval: {train_config.get('checkpoint_iters', 1000)}")
        print(f"Validation interval: {train_config.get('num_valid_iters', 10)}")
        print(f"MixUp probability: {config['dataloader'].get('mixup_prob', 0.5)}")
        print(f"MixUp alpha: {config['dataloader'].get('mixup_alpha')}")
        print(f"Loss function: {config['train']['loss']['name']}")
        print(f"Device: {device}")
        print(f"Distributed training: {is_distributed}")        
        if is_distributed:
            print(f"World size: {world_size}")
            print(f"Backend: {config['distributed'].get('backend', 'nccl')}")
        print("="*60 + "\n")
    
    if args.dry_run:
        if rank==0:
            print("DRY RUN COMPLETE - Exiting without training")
        cleanup_distributed()
        return
    
    # Training loop
    if rank == 0:
        print("Starting training loop!")

    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    last_iepoch = -1
    end_iteration = start_iteration + train_config['num_iterations']
    epoch = int( float(start_iteration) / float(train_iters_per_epoch) )
    if train_sampler is not None:
        train_sampler.set_epoch( epoch )
        valid_sampler.set_epoch( int(epoch) * 1000 )

    for iteration in range(start_iteration,end_iteration):

        # Set epoch for distributed sampler
        if train_sampler is not None:
            current_epoch = int(epoch)
            if current_epoch != last_iepoch:
                train_sampler.set_epoch(current_epoch)
                last_iepoch = current_epoch

        siren.train()
        optimizer.zero_grad()
    
        tstart_dataprep = time.time()

        try:
            if train_config['freeze_batch']:
                if iteration==0:
                    batch = next(train_iter)
                else:
                    pass # intentionally not changing data
            else:
                batch = next(train_iter)
        except StopIteration:
            # Properly handle end of epoch
            if rank==0:
                print(f"Training epoch {int(epoch)} completed, starting new epoch")

            # Update epoch for samplers before creating new iterator
            epoch = float(iteration + 1) / float(train_iters_per_epoch)
            if train_sampler is not None:
                current_epoch = int(epoch)
                train_sampler.set_epoch(current_epoch)
                if rank==0:
                    print(f"Set train_sampler epoch to {current_epoch}")

            train_iter = iter(train_loader)
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

        effective_batch_size = config['dataloader'].get('batchsize')
        if is_distributed and not config['distributed'].get('scale_batch_size', False):
            effective_batch_size = max(1, effective_batch_size // world_size)

        if Nb!=effective_batch_size:
            if rank==0:
                print(f"UNEXPECTED BATCHSIZE ({Nb} vs {effective_batch_size}), SKIPPING THIS ITERATION")
            # Skip this iteration and continue to next
            continue


        q_feat = batch['planecharge_normalized'].to(device)
        mask   = batch['mask'].to(device)
        n_voxels = batch['n_voxels'].to(device)
        start_per_batch = torch.zeros( effective_batch_size, dtype=torch.int64 ).to(device)
        end_per_batch   = torch.zeros( effective_batch_size, dtype=torch.int64 ).to(device)
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
            if rank==0:
                print("INPUTS ==================")
                print("vox_feat.shape=",vox_feat.shape," from (Nb x Nv,Npmt,Nk)=",(Nbv,Npmt,K))
                print("q.shape=",q.shape)
                if debug:
                    print("-------q dump -------------------------------")
                    print(q.reshape((Nb,Nv,Npmt))[0,:,0])
                    print("---------------------------------------------")

        pe_per_voxel = siren(vox_feat, q)
        if rank==0 and debug:
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
        if rank==0:
            print("forward time: ",dt_forward)
            if debug:           
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

        if rank==0 and debug:
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
        next_lr = get_learning_rate( 
            epoch, 
            lr_config['warmup_nepochs'], 
            lr_config['cosine_nepochs'],
            lr_config['warmup_lr'],
            lr_config['max'], 
            lr_config['min'] 
        )
        # update LR
        optimizer.param_groups[1]["lr"] = next_lr
        optimizer.param_groups[0]["lr"] = next_lr*1.0e-5 # LY learning rate

        dt_backward = time.time()-dt_backward

        if rank==0:
            print("backward time: ",dt_backward," secs")

        if iteration>start_iteration and iteration%int(config['train'].get('checkpoint_iters'))==0:
            if rank==0:
                print('save checkpoint')
            save_checkpoint(
                {'siren':siren},
                {'siren':optimizer},
                iteration,
                config,
                config['train'].get('checkpoint_folder')+"/checkpoint_iteration_%08d.pt"%(iteration),
                rank
            )

        if iteration>0 and iteration%int(config['train'].get('num_valid_iters'))==0:
            with torch.no_grad():
                valid_info_dict, valid_iter = run_validation(
                    config, iteration, epoch, siren,
                    loss_fn_valid, valid_iter, valid_loader, valid_sampler,
                    device, pmtpos, rank, is_train_eval=False )

                train_info_dict, train_iter = run_validation(
                    config, iteration, epoch, siren,
                    loss_fn_train, train_iter, train_loader, train_sampler,
                    device, pmtpos, rank, is_train_eval=True )
        
                if rank==0 and config['logger'].get('use_wandb'):
                    ly_value = siren.module.get_light_yield().cpu().item() if isinstance(siren, DDP) else siren.get_light_yield().cpu().item()
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
                        "lightyield":ly_value,
                        "epoch":epoch}
                    wandb.log(for_wandb, step=iteration)

        
    if rank==0:
        print("FINISHED ITERATION LOOP!")
        save_checkpoint(
            {'siren':siren},
            {'siren':optimizer},
            iteration,
            config,
            config['train'].get('checkpoint_folder')+"/checkpoint_iteration_%08d.pt"%(end_iteration),
            rank
        )
    
    # Close W&B run
    if wandb_run:
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
