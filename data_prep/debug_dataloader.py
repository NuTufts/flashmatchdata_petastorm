import os, sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import traceback

sys.path.append(os.path.dirname(__file__))
from read_flashmatch_hdf5 import FlashMatchVoxelDataset

def debug_dataset(filelist_path):
    """Debug the dataset to find problematic entries"""
    
    # Create dataset with single worker to make debugging easier
    dataset = FlashMatchVoxelDataset(
        hdf5_files=filelist_path,
        max_voxels=500,
        load_to_memory=False
    )
    
    print(f"Total dataset size: {len(dataset)}")
    print("\nChecking individual entries...")
    
    # Check first few entries individually
    problem_entries = []
    expected_shape = None
    
    for i in range(min(100, len(dataset))):
        try:
            entry = dataset[i]
            
            # Check shapes
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    if key == 'features':
                        current_shape = value.shape
                        if expected_shape is None:
                            expected_shape = current_shape
                            print(f"Entry {i}: features shape = {current_shape} (setting as expected)")
                        elif current_shape != expected_shape:
                            print(f"ERROR at entry {i}: features shape = {current_shape}, expected = {expected_shape}")
                            problem_entries.append(i)
                            
                            # Get more info about this entry
                            file_idx, entry_idx = dataset.file_entries[i]
                            print(f"  File index: {file_idx}, Entry index: {entry_idx}")
                            print(f"  File: {dataset.hdf5_files[file_idx]}")
                            
                            # Check the raw data
                            print(f"  planecharge shape: {entry['planecharge'].shape}")
                            print(f"  avepos shape: {entry['avepos'].shape}")
                            print(f"  n_voxels: {entry['n_voxels']}")
                            break
                            
        except Exception as e:
            print(f"Error at entry {i}: {e}")
            traceback.print_exc()
            problem_entries.append(i)
    
    if problem_entries:
        print(f"\nFound {len(problem_entries)} problematic entries: {problem_entries[:10]}")
    else:
        print("\nNo problems found in individual entry access")
    
    # Now try with DataLoader
    print("\nTesting with DataLoader (batch_size=32)...")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,  # Don't shuffle to make debugging easier
        num_workers=0,  # Single process for debugging
        pin_memory=False
    )
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                print(f"First batch successful, features shape: {batch['features'].shape}")
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx} batches...")
            if batch_idx >= 10:  # Just test first 10 batches
                break
    except Exception as e:
        print(f"\nError in batch {batch_idx}:")
        print(str(e))
        print("\nTrying to identify the problematic sample in the batch...")
        
        # Get the indices for this batch
        start_idx = batch_idx * 32
        end_idx = min(start_idx + 32, len(dataset))
        
        for i in range(start_idx, end_idx):
            try:
                entry = dataset[i]
                if 'features' in entry:
                    print(f"  Entry {i}: features shape = {entry['features'].shape}")
            except Exception as e2:
                print(f"  Entry {i}: ERROR - {e2}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_dataloader.py <filelist>")
        sys.exit(1)
    
    filelist = sys.argv[1]
    debug_dataset(filelist)