import os,sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import time
from tqdm import tqdm

from read_flashmatch_hdf5 import FlashMatchVoxelDataset

hdf5_filelist = sys.argv[1]

dataset = FlashMatchVoxelDataset(
    hdf5_files=hdf5_filelist,
    max_voxels=500,
    load_to_memory=False
)
    
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Calculate total batches for progress bar
total_batches = len(dataloader)
print(f"Dataset size: {len(dataset)} entries")
print(f"Batch size: 32")
print(f"Total batches: {total_batches}")
print(f"Number of workers: 4")
print("-" * 50)

# Training loop with progress bar
print("Starting epoch read test...")
tstart = time.time()

for epoch in range(1):
    # Create progress bar for batches
    pbar = tqdm(dataloader, 
                desc=f"Epoch {epoch+1}",
                total=total_batches,
                unit="batch",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        features = batch['features']  # (batch, max_voxels, 6)
        mask = batch['mask']  # (batch, max_voxels)
        
        if torch.cuda.is_available():
            features = features.cuda()
            mask = mask.cuda()
        
        # Update progress bar with additional stats every 100 batches
        if batch_idx % 100 == 0:
            elapsed = time.time() - tstart
            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            samples_per_sec = (batch_idx + 1) * 32 / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'batch/s': f'{batches_per_sec:.2f}',
                'samples/s': f'{samples_per_sec:.0f}'
            })

dt_epoch_read = time.time()-tstart
print("-" * 50)
print(f"✓ Completed 1 epoch")
print(f"  Total time: {dt_epoch_read:.2f} seconds")
print(f"  Average time per batch: {dt_epoch_read/total_batches:.4f} seconds")
print(f"  Average throughput: {len(dataset)/dt_epoch_read:.1f} samples/second")