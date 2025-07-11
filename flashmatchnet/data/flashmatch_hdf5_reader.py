#!/usr/bin/env python
import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

try:
    import MinkowskiEngine as ME
except:
    print("Warning: MinkowskiEngine not available, sparse tensor functions will not work")
    ME = None


class FlashMatchHDF5Dataset(Dataset):
    """
    PyTorch Dataset for reading FlashMatch HDF5 training data.
    """
    
    # Define column names for the HDF5 file
    COLUMNS = OrderedDict([
        ('sourcefile', np.str),
        ('run', np.int32),
        ('subrun', np.int32), 
        ('event', np.int32),
        ('matchindex', np.int32),
        ('ancestorid', np.int32),
        ('coord', np.int64),      # Variable length array
        ('feat', np.float32),     # Variable length array  
        ('flashpe', np.float32),  # Variable length array
    ])
    
    def __init__(self, hdf5_files, transform=None, load_into_memory=False):
        """
        Args:
            hdf5_files: Single HDF5 file path or list of HDF5 file paths
            transform: Optional transform to apply to data
            load_into_memory: If True, load all data into memory at initialization
        """
        if isinstance(hdf5_files, str):
            hdf5_files = [hdf5_files]
            
        self.hdf5_files = hdf5_files
        self.transform = transform
        self.load_into_memory = load_into_memory
        
        # Build index of entries across all files
        self._build_index()
        
        # Load data into memory if requested
        if self.load_into_memory:
            self._load_all_data()
            
    def _build_index(self):
        """Build an index mapping dataset indices to (file_idx, entry_idx) pairs."""
        self.file_handles = []
        self.entry_index = []
        self.total_entries = 0
        
        for file_idx, filepath in enumerate(self.hdf5_files):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"HDF5 file not found: {filepath}")
                
            h5file = h5py.File(filepath, 'r')
            self.file_handles.append(h5file)
            
            # Get number of entries in this file
            n_entries = len(h5file['run'])
            
            # Add entries to index
            for entry_idx in range(n_entries):
                self.entry_index.append((file_idx, entry_idx))
                
            self.total_entries += n_entries
            
        print(f"Loaded {self.total_entries} entries from {len(self.hdf5_files)} files")
        
    def _load_all_data(self):
        """Load all data into memory."""
        print("Loading all data into memory...")
        self.data_cache = []
        
        for idx in range(len(self)):
            self.data_cache.append(self._get_entry(idx))
            
        print("Data loading complete")
        
    def _get_entry(self, idx):
        """Get a single entry from the dataset."""
        file_idx, entry_idx = self.entry_index[idx]
        h5file = self.file_handles[file_idx]
        
        # Read data for this entry
        data = {}
        data['sourcefile'] = h5file['sourcefile'][entry_idx].decode('utf-8')
        data['run'] = int(h5file['run'][entry_idx])
        data['subrun'] = int(h5file['subrun'][entry_idx])
        data['event'] = int(h5file['event'][entry_idx])
        data['matchindex'] = int(h5file['matchindex'][entry_idx])
        data['ancestorid'] = int(h5file['ancestorid'][entry_idx])
        
        # Variable length arrays
        coord_flat = h5file['coord'][entry_idx]
        feat_flat = h5file['feat'][entry_idx]
        flashpe_flat = h5file['flashpe'][entry_idx]
        
        # Reshape arrays
        data['coord'] = coord_flat.reshape(-1, 3)
        data['feat'] = feat_flat.reshape(-1, 3)
        data['flashpe'] = flashpe_flat.reshape(-1, 32)  # 32 PMTs
        
        return data
        
    def __len__(self):
        return self.total_entries
        
    def __getitem__(self, idx):
        if self.load_into_memory:
            data = self.data_cache[idx]
        else:
            data = self._get_entry(idx)
            
        if self.transform is not None:
            data = self.transform(data)
            
        return data
        
    def close(self):
        """Close all HDF5 file handles."""
        for h5file in self.file_handles:
            h5file.close()
            

def flashmatch_collate_fn(batch):
    """
    Custom collate function for FlashMatch data that handles variable-length tensors.
    
    Args:
        batch: List of data dictionaries from FlashMatchHDF5Dataset
        
    Returns:
        Dictionary with batched data
    """
    # Separate sparse (variable length) and dense data
    dense_keys = ['run', 'subrun', 'event', 'matchindex', 'ancestorid']
    
    # Collate dense data
    collated = {}
    for key in dense_keys:
        collated[key] = torch.tensor([item[key] for item in batch])
        
    # Handle source files
    collated['sourcefile'] = [item['sourcefile'] for item in batch]
    
    # Collate flashpe (can be stacked since all have same shape)
    collated['flashpe'] = torch.stack([torch.tensor(item['flashpe']) for item in batch])
    
    # Handle variable-length coordinate and feature data
    coords_list = []
    feats_list = []
    batch_indices = []
    
    for batch_idx, item in enumerate(batch):
        n_points = len(item['coord'])
        coords_list.append(torch.tensor(item['coord']))
        feats_list.append(torch.tensor(item['feat']))
        batch_indices.append(torch.full((n_points,), batch_idx, dtype=torch.long))
        
    # Concatenate all coordinates and features
    collated['coords'] = torch.cat(coords_list, dim=0)
    collated['feats'] = torch.cat(feats_list, dim=0)
    collated['batch_indices'] = torch.cat(batch_indices, dim=0)
    
    return collated


def flashmatch_collate_sparse_fn(batch):
    """
    Collate function that creates MinkowskiEngine sparse tensors.
    
    Args:
        batch: List of data dictionaries from FlashMatchHDF5Dataset
        
    Returns:
        Dictionary with batched data including sparse tensors
    """
    if ME is None:
        raise ImportError("MinkowskiEngine is required for sparse tensor collation")
        
    # First use regular collate
    collated = flashmatch_collate_fn(batch)
    
    # Create sparse tensor coordinates (add batch index as first dimension)
    sparse_coords = torch.cat([
        collated['batch_indices'].unsqueeze(1),
        collated['coords']
    ], dim=1)
    
    # Create sparse tensor
    collated['sparse_tensor'] = ME.SparseTensor(
        features=collated['feats'],
        coordinates=sparse_coords,
        device='cpu'
    )
    
    return collated


class FlashMatchTransform:
    """
    Transform class for FlashMatch data augmentation and preprocessing.
    """
    
    def __init__(self, voxel_size=5.0, augment=False):
        """
        Args:
            voxel_size: Size of voxels in cm
            augment: Whether to apply data augmentation
        """
        self.voxel_size = voxel_size
        self.augment = augment
        
    def __call__(self, data):
        """
        Apply transformations to data.
        
        Args:
            data: Dictionary from FlashMatchHDF5Dataset
            
        Returns:
            Transformed data dictionary
        """
        # Convert to tensors if not already
        if not isinstance(data['coord'], torch.Tensor):
            data['coord'] = torch.tensor(data['coord'], dtype=torch.long)
        if not isinstance(data['feat'], torch.Tensor):
            data['feat'] = torch.tensor(data['feat'], dtype=torch.float32)
        if not isinstance(data['flashpe'], torch.Tensor):
            data['flashpe'] = torch.tensor(data['flashpe'], dtype=torch.float32)
            
        if self.augment:
            # Add augmentation here if needed
            # For example: random flips, rotations, noise, etc.
            pass
            
        return data


def create_flashmatch_dataloader(hdf5_files, batch_size=32, shuffle=True, 
                               num_workers=4, use_sparse=False, **kwargs):
    """
    Create a DataLoader for FlashMatch HDF5 data.
    
    Args:
        hdf5_files: Single file path or list of file paths
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        use_sparse: Whether to use sparse tensor collation
        **kwargs: Additional arguments passed to FlashMatchHDF5Dataset
        
    Returns:
        DataLoader instance
    """
    dataset = FlashMatchHDF5Dataset(hdf5_files, **kwargs)
    
    collate_fn = flashmatch_collate_sparse_fn if use_sparse else flashmatch_collate_fn
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FlashMatch HDF5 reader')
    parser.add_argument('hdf5_file', type=str, help='Path to HDF5 file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--use-sparse', action='store_true', help='Use sparse tensors')
    
    args = parser.parse_args()
    
    # Create dataloader
    dataloader = create_flashmatch_dataloader(
        args.hdf5_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        use_sparse=args.use_sparse
    )
    
    # Test iteration
    print(f"\nTesting dataloader with batch_size={args.batch_size}")
    print("="*50)
    
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Run: {batch['run'].tolist()}")
        print(f"  Subrun: {batch['subrun'].tolist()}")
        print(f"  Event: {batch['event'].tolist()}")
        print(f"  FlashPE shape: {batch['flashpe'].shape}")
        print(f"  Coords shape: {batch['coords'].shape}")
        print(f"  Feats shape: {batch['feats'].shape}")
        
        if args.use_sparse:
            print(f"  Sparse tensor: {batch['sparse_tensor']}")
            
        if i >= 2:  # Only show first 3 batches
            break
            
    print("\nDataloader test complete!")