#!/usr/bin/env python3
"""
FlashMatch MixUp Data Augmentation

Implements MixUp augmentation for flash matching data where:
- PMT observations are weighted-summed: pe_mixed = alpha * pe_a + (1-alpha) * pe_b
- Voxel features are concatenated with weighted planecharge: concat([voxels_a * alpha, voxels_b * (1-alpha)])
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple
import random

class MixUpFlashMatchDataset(Dataset):
    """
    Wrapper dataset that applies MixUp augmentation to FlashMatch data
    
    MixUp combines two samples:
    - Target PMT values are linearly combined
    - Voxel features are concatenated (with weighted planecharge)
    """
    
    def __init__(self, 
                 base_dataset,
                 mixup_prob: float = 0.5,
                 alpha: float = 1.0,
                 max_total_voxels: Optional[int] = None):
        """
        Args:
            base_dataset: The base FlashMatchVoxelDataset
            mixup_prob: Probability of applying mixup (0 = never, 1 = always)
            alpha: Beta distribution parameter for sampling mixing coefficient
                   Higher alpha = mixing coefficients closer to 0.5
                   alpha=1.0 gives uniform distribution
            max_total_voxels: Maximum total voxels after mixing (will truncate if exceeded)
                             If None, uses 2 * base_dataset.max_voxels
        """
        self.base_dataset = base_dataset
        self.mixup_prob = mixup_prob
        self.alpha = alpha
        self.max_total_voxels = max_total_voxels or (2 * base_dataset.max_voxels)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def sample_lambda(self) -> float:
        """Sample mixing coefficient from Beta distribution"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        return lam
    
    def mixup_samples(self, sample1: Dict, sample2: Dict, lam: float) -> Dict:
        """
        Apply MixUp to two samples
        
        Args:
            sample1: First sample dictionary
            sample2: Second sample dictionary  
            lam: Mixing coefficient (weight for sample1)
            
        Returns:
            Mixed sample dictionary
        """
        # Extract voxel counts
        n_voxels1 = sample1['n_voxels'].item()
        n_voxels2 = sample2['n_voxels'].item()
        n_total = n_voxels1 + n_voxels2
        
        # Mix PMT observations (weighted sum)
        mixed_pmt = lam * sample1['observed_pe_per_pmt'] + (1 - lam) * sample2['observed_pe_per_pmt']
        mixed_total_pe = lam * sample1['observed_total_pe'] + (1 - lam) * sample2['observed_total_pe']
        
        # For predicted values (if doing supervised learning on predictions)
        mixed_pred_pmt = lam * sample1['predicted_pe_per_pmt'] + (1 - lam) * sample2['predicted_pe_per_pmt']
        mixed_pred_total = lam * sample1['predicted_total_pe'] + (1 - lam) * sample2['predicted_total_pe']
        
        # Extract valid voxels (before padding)
        planecharge1 = sample1['planecharge'][:n_voxels1]  # (n_voxels1, 3)
        planecharge2 = sample2['planecharge'][:n_voxels2]  # (n_voxels2, 3)
        
        avepos1 = sample1['avepos'][:n_voxels1]  # (n_voxels1, 3)
        avepos2 = sample2['avepos'][:n_voxels2]  # (n_voxels2, 3)
        
        centers1 = sample1['centers'][:n_voxels1]
        centers2 = sample2['centers'][:n_voxels2]
        
        indices1 = sample1['indices'][:n_voxels1]
        indices2 = sample2['indices'][:n_voxels2]
        
        # Apply weights to planecharge
        weighted_planecharge1 = planecharge1 * lam
        weighted_planecharge2 = planecharge2 * (1 - lam)
        
        # Concatenate voxel data
        mixed_planecharge = torch.cat([weighted_planecharge1, weighted_planecharge2], dim=0)
        mixed_avepos = torch.cat([avepos1, avepos2], dim=0)
        mixed_centers = torch.cat([centers1, centers2], dim=0)
        mixed_indices = torch.cat([indices1, indices2], dim=0)
        
        # Handle case where total voxels exceed maximum
        if n_total > self.max_total_voxels:
            # Randomly sample voxels to keep
            indices_to_keep = torch.randperm(n_total)[:self.max_total_voxels]
            mixed_planecharge = mixed_planecharge[indices_to_keep]
            mixed_avepos = mixed_avepos[indices_to_keep]
            mixed_centers = mixed_centers[indices_to_keep]
            mixed_indices = mixed_indices[indices_to_keep]
            n_total = self.max_total_voxels
        
        # Pad to max_total_voxels
        pad_size = self.max_total_voxels - n_total
        if pad_size > 0:
            mixed_planecharge = torch.nn.functional.pad(mixed_planecharge, (0, 0, 0, pad_size))
            mixed_avepos = torch.nn.functional.pad(mixed_avepos, (0, 0, 0, pad_size))
            mixed_centers = torch.nn.functional.pad(mixed_centers, (0, 0, 0, pad_size))
            mixed_indices = torch.nn.functional.pad(mixed_indices, (0, 0, 0, pad_size))
        
        # Create mask for valid voxels
        mask = torch.zeros(self.max_total_voxels)
        mask[:n_total] = 1.0
        
        # Recreate features (concatenate planecharge and normalized positions)
        mixed_features = torch.cat([
            mixed_planecharge,  # Already weighted
            mixed_avepos / 1000.0,  # Normalize positions
        ], dim=1)
        
        # Create mixed sample
        mixed_sample = {
            'planecharge': mixed_planecharge,
            'indices': mixed_indices,
            'avepos': mixed_avepos,
            'centers': mixed_centers,
            'mask': mask,
            'n_voxels': torch.tensor(n_total, dtype=torch.int64),
            'features': mixed_features,
            # Mixed PMT data
            'observed_pe_per_pmt': mixed_pmt,
            'predicted_pe_per_pmt': mixed_pred_pmt,
            'observed_total_pe': mixed_total_pe,
            'predicted_total_pe': mixed_pred_total,
            # Store mixing info
            'mixup_lambda': torch.tensor(lam, dtype=torch.float32),
            'mixup_applied': torch.tensor(1, dtype=torch.int64),
            # We lose individual event metadata in mixing
            'match_type': torch.tensor(-2, dtype=torch.int64),  # -2 indicates mixed
            'run':sample1['run'],
            'subrun':sample1['subrun'],
            'event':sample1['event'],
            'match_index':sample1['match_index'],
        }
        
        return mixed_sample
    
    def __getitem__(self, idx):
        """Get item with potential MixUp augmentation"""
        # Get first sample
        sample1 = self.base_dataset[idx]
        
        # Decide whether to apply mixup
        if random.random() < self.mixup_prob:
            # Randomly select another sample to mix with
            idx2 = random.randint(0, len(self.base_dataset) - 1)
            sample2 = self.base_dataset[idx2]
            
            # Sample mixing coefficient
            lam = self.sample_lambda()
            
            # Apply mixup
            return self.mixup_samples(sample1, sample2, lam)
        else:
            # No mixup - need to ensure consistent tensor sizes
            # Adjust mask and features for potentially larger max_total_voxels
            if self.max_total_voxels > self.base_dataset.max_voxels:
                n_voxels = sample1['n_voxels'].item()
                
                # Pad all tensors to max_total_voxels
                pad_size = self.max_total_voxels - self.base_dataset.max_voxels
                
                sample1['planecharge'] = torch.nn.functional.pad(sample1['planecharge'], (0, 0, 0, pad_size))
                sample1['avepos'] = torch.nn.functional.pad(sample1['avepos'], (0, 0, 0, pad_size))
                sample1['centers'] = torch.nn.functional.pad(sample1['centers'], (0, 0, 0, pad_size))
                sample1['indices'] = torch.nn.functional.pad(sample1['indices'], (0, 0, 0, pad_size))
                sample1['features'] = torch.nn.functional.pad(sample1['features'], (0, 0, 0, pad_size))
                
                # Recreate mask
                mask = torch.zeros(self.max_total_voxels)
                mask[:n_voxels] = 1.0
                sample1['mask'] = mask
            
            # Add mixup metadata
            sample1['mixup_lambda'] = torch.tensor(1.0, dtype=torch.float32)
            sample1['mixup_applied'] = torch.tensor(0, dtype=torch.int64)
            
            return sample1


def mixup_collate_fn(batch):
    """
    Custom collate function for batching mixup samples
    Handles variable max_voxels sizes
    """
    # Find the maximum number of voxels in this batch
    max_voxels_in_batch = max(sample['planecharge'].shape[0] for sample in batch)
    
    # Pad all samples to have the same number of voxels
    for sample in batch:
        current_voxels = sample['planecharge'].shape[0]
        if current_voxels < max_voxels_in_batch:
            pad_size = max_voxels_in_batch - current_voxels
            
            # Pad all voxel-related tensors
            sample['planecharge'] = torch.nn.functional.pad(sample['planecharge'], (0, 0, 0, pad_size))
            sample['avepos'] = torch.nn.functional.pad(sample['avepos'], (0, 0, 0, pad_size))
            sample['centers'] = torch.nn.functional.pad(sample['centers'], (0, 0, 0, pad_size))
            sample['indices'] = torch.nn.functional.pad(sample['indices'], (0, 0, 0, pad_size))
            sample['features'] = torch.nn.functional.pad(sample['features'], (0, 0, 0, pad_size))
            
            # Update mask
            n_voxels = sample['n_voxels'].item()
            mask = torch.zeros(max_voxels_in_batch)
            mask[:n_voxels] = 1.0
            sample['mask'] = mask
    
    # Now use default collate since all tensors have same shape
    #print(batch)
    return torch.utils.data.dataloader.default_collate(batch)


def create_mixup_dataloader(base_dataset,
                           batch_size: int = 32,
                           mixup_prob: float = 0.5,
                           alpha: float = 1.0,
                           max_total_voxels: Optional[int] = None,
                           shuffle: bool = True,
                           num_workers: int = 4,
                           **kwargs):
    """
    Create a DataLoader with MixUp augmentation
    
    Args:
        base_dataset: Base FlashMatchVoxelDataset
        batch_size: Batch size
        mixup_prob: Probability of applying mixup to each sample
        alpha: Beta distribution parameter for mixing coefficient
        max_total_voxels: Maximum voxels after mixing
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader with MixUp augmentation
    """
    mixup_dataset = MixUpFlashMatchDataset(
        base_dataset=base_dataset,
        mixup_prob=mixup_prob,
        alpha=alpha,
        max_total_voxels=max_total_voxels
    )
    
    dataloader = DataLoader(
        mixup_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mixup_collate_fn,
        **kwargs
    )
    
    return dataloader


if __name__ == "__main__":
    """Example usage of MixUp augmentation"""
    import sys
    sys.path.append('.')
    from read_flashmatch_hdf5 import FlashMatchVoxelDataset
    
    # Create base dataset
    print("Creating base dataset...")
    base_dataset = FlashMatchVoxelDataset(
        hdf5_files='filelist.txt',
        max_voxels=500,
        load_to_memory=False
    )
    
    # Create MixUp dataset
    print("\nCreating MixUp dataset...")
    mixup_dataset = MixUpFlashMatchDataset(
        base_dataset=base_dataset,
        mixup_prob=1.0,  # Always apply mixup for testing
        alpha=1.0,
        max_total_voxels=1000  # Allow up to 1000 voxels after mixing
    )
    
    # Test single sample
    print("\nTesting single MixUp sample...")
    mixed_sample = mixup_dataset[0]
    
    print(f"Mixed sample shapes:")
    print(f"  planecharge: {mixed_sample['planecharge'].shape}")
    print(f"  features: {mixed_sample['features'].shape}")
    print(f"  observed_pe_per_pmt: {mixed_sample['observed_pe_per_pmt'].shape}")
    print(f"  n_voxels: {mixed_sample['n_voxels'].item()}")
    print(f"  mixup_lambda: {mixed_sample['mixup_lambda'].item():.3f}")
    print(f"  mixup_applied: {mixed_sample['mixup_applied'].item()}")
    
    # Test with DataLoader
    print("\nTesting DataLoader with MixUp...")
    dataloader = create_mixup_dataloader(
        base_dataset,
        batch_size=4,
        mixup_prob=0.5,  # 50% chance of mixup
        alpha=1.0,
        max_total_voxels=1000,
        shuffle=True,
        num_workers=0  # Single process for testing
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  features: {batch['features'].shape}")
    print(f"  observed_pe_per_pmt: {batch['observed_pe_per_pmt'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  mixup_applied: {batch['mixup_applied']}")
    print(f"  mixup_lambdas: {batch['mixup_lambda']}")
    
    # Check that mixed samples have reasonable values
    for i in range(len(batch['mixup_applied'])):
        if batch['mixup_applied'][i] == 1:
            print(f"\nSample {i} was mixed:")
            print(f"  Lambda: {batch['mixup_lambda'][i]:.3f}")
            print(f"  N voxels: {batch['n_voxels'][i]}")
            print(f"  Total PE: {batch['observed_total_pe'][i]:.1f}")