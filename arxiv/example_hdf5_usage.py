#!/usr/bin/env python
"""
Example script showing how to use the new HDF5 reader and writer for FlashMatch data.

This script demonstrates:
1. How to create HDF5 training data from ROOT files
2. How to read and iterate over HDF5 training data
3. Basic data inspection and visualization
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py

# Import our new HDF5 modules
from flashmatch_hdf5_writer import main as writer_main
from flashmatch_hdf5_reader import FlashMatchHDF5Dataset, create_flashmatch_dataloader, FlashMatchTransform

def create_sample_data(output_file):
    """
    Create sample HDF5 data for testing.
    This would normally use the flashmatch_hdf5_writer.py with real ROOT files.
    """
    print("Creating sample HDF5 file for demonstration...")
    
    # Create some dummy data that matches our schema
    n_entries = 100
    
    with h5py.File(output_file, 'w') as f:
        # Scalar datasets
        f.create_dataset('sourcefile', data=np.array([f'test_file_{i%10}.root' for i in range(n_entries)], dtype='S256'))
        f.create_dataset('run', data=np.random.randint(1000, 2000, n_entries, dtype=np.int32))
        f.create_dataset('subrun', data=np.random.randint(0, 100, n_entries, dtype=np.int32))
        f.create_dataset('event', data=np.arange(n_entries, dtype=np.int32))
        f.create_dataset('matchindex', data=np.random.randint(0, 5, n_entries, dtype=np.int32))
        f.create_dataset('ancestorid', data=np.random.randint(1, 1000, n_entries, dtype=np.int32))
        
        # Variable length datasets
        coord_dtype = h5py.special_dtype(vlen=np.dtype('int64'))
        feat_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
        flashpe_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
        
        coord_dset = f.create_dataset('coord', (n_entries,), dtype=coord_dtype)
        feat_dset = f.create_dataset('feat', (n_entries,), dtype=feat_dtype)
        flashpe_dset = f.create_dataset('flashpe', (n_entries,), dtype=flashpe_dtype)
        
        for i in range(n_entries):
            # Random number of voxels per entry (between 50 and 500)
            n_voxels = np.random.randint(50, 500)
            
            # Random 3D coordinates (simulating voxel indices)
            coords = np.random.randint(0, 50, (n_voxels, 3)).astype(np.int64)
            
            # Random features (charge per plane)
            feats = np.random.exponential(1000, (n_voxels, 3)).astype(np.float32)
            
            # Random flash PE values for 32 PMTs
            flashpe = np.random.exponential(10, (1, 32)).astype(np.float32)
            
            # Store flattened arrays
            coord_dset[i] = coords.flatten()
            feat_dset[i] = feats.flatten()
            flashpe_dset[i] = flashpe.flatten()
            
        # Store metadata
        f.attrs['num_entries'] = n_entries
        f.attrs['npmts'] = 32
        f.attrs['voxel_size_cm'] = 5.0
        
    print(f"Created sample data file: {output_file} with {n_entries} entries")

def inspect_hdf5_file(hdf5_file):
    """
    Inspect the contents of an HDF5 file.
    """
    print(f"\n=== Inspecting HDF5 file: {hdf5_file} ===")
    
    with h5py.File(hdf5_file, 'r') as f:
        print("Datasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: {dataset.shape}, dtype: {dataset.dtype}")
            
        print("\nAttributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
            
        # Show some sample data
        print(f"\nSample data from first entry:")
        print(f"  Run: {f['run'][0]}")
        print(f"  Event: {f['event'][0]}")
        print(f"  Coord shape: {len(f['coord'][0])//3} voxels")
        print(f"  FlashPE sum: {f['flashpe'][0].sum():.2f}")

def test_dataset_reader(hdf5_file):
    """
    Test the FlashMatchHDF5Dataset class.
    """
    print(f"\n=== Testing Dataset Reader ===")
    
    # Create dataset
    dataset = FlashMatchHDF5Dataset(hdf5_file)
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting a single item
    item = dataset[0]
    print(f"Sample item keys: {item.keys()}")
    print(f"Coord shape: {item['coord'].shape}")
    print(f"Feat shape: {item['feat'].shape}")
    print(f"FlashPE shape: {item['flashpe'].shape}")
    print(f"Run: {item['run']}, Event: {item['event']}")
    
    # Close dataset
    dataset.close()

def test_dataloader(hdf5_file):
    """
    Test the DataLoader functionality.
    """
    print(f"\n=== Testing DataLoader ===")
    
    # Create transform
    transform = FlashMatchTransform(voxel_size=5.0, augment=False)
    
    # Create dataloader
    dataloader = create_flashmatch_dataloader(
        hdf5_file,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for single-threaded debugging
        use_sparse=False,
        transform=transform
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Batch size: {len(batch['run'])}")
        print(f"  Run: {batch['run'].tolist()}")
        print(f"  FlashPE shape: {batch['flashpe'].shape}")
        print(f"  Total coords: {batch['coords'].shape}")
        print(f"  Total feats: {batch['feats'].shape}")
        print(f"  Batch indices range: {batch['batch_indices'].min().item()} - {batch['batch_indices'].max().item()}")
        
        if i >= 2:  # Only show first 3 batches
            break

def test_sparse_dataloader(hdf5_file):
    """
    Test the sparse tensor DataLoader functionality.
    """
    print(f"\n=== Testing Sparse DataLoader ===")
    
    try:
        import MinkowskiEngine as ME
    except ImportError:
        print("MinkowskiEngine not available, skipping sparse tensor test")
        return
    
    # Create dataloader with sparse tensors
    dataloader = create_flashmatch_dataloader(
        hdf5_file,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        use_sparse=True
    )
    
    # Test iteration
    for i, batch in enumerate(dataloader):
        print(f"\nSparse Batch {i}:")
        print(f"  Sparse tensor: {batch['sparse_tensor']}")
        print(f"  Sparse coords shape: {batch['sparse_tensor'].coordinates.shape}")
        print(f"  Sparse feats shape: {batch['sparse_tensor'].features.shape}")
        
        if i >= 1:  # Only show first 2 batches
            break

def main():
    parser = argparse.ArgumentParser(description='Example usage of FlashMatch HDF5 reader/writer')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample HDF5 data for testing')
    parser.add_argument('--hdf5-file', type=str, default='sample_flashmatch_data.h5',
                        help='HDF5 file to use for testing')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect HDF5 file contents')
    parser.add_argument('--test-dataset', action='store_true',
                        help='Test dataset reader')
    parser.add_argument('--test-dataloader', action='store_true',
                        help='Test dataloader')
    parser.add_argument('--test-sparse', action='store_true',
                        help='Test sparse tensor dataloader')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample or args.all:
        create_sample_data(args.hdf5_file)
    
    # Check if file exists
    if not os.path.exists(args.hdf5_file):
        print(f"Error: HDF5 file {args.hdf5_file} does not exist.")
        print("Use --create-sample to create sample data first.")
        return
    
    # Run tests
    if args.inspect or args.all:
        inspect_hdf5_file(args.hdf5_file)
        
    if args.test_dataset or args.all:
        test_dataset_reader(args.hdf5_file)
        
    if args.test_dataloader or args.all:
        test_dataloader(args.hdf5_file)
        
    if args.test_sparse or args.all:
        test_sparse_dataloader(args.hdf5_file)
    
    print("\n=== Example Usage Summary ===")
    print("1. To create HDF5 data from ROOT files:")
    print("   python flashmatch_hdf5_writer.py -o output.h5 -lcv truth.root -mc mcinfo.root -op opreco.root")
    print("")
    print("2. To use in training scripts:")
    print("   from flashmatch_hdf5_reader import create_flashmatch_dataloader")
    print("   dataloader = create_flashmatch_dataloader('data.h5', batch_size=32)")
    print("")
    print("3. To convert existing training scripts:")
    print("   - Replace make_dataloader with create_flashmatch_dataloader")
    print("   - Update file paths to point to HDF5 files instead of Petastorm URLs")
    print("   - See train_mlp_hdf5.py for a complete example")

if __name__ == "__main__":
    main()