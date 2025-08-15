#!/usr/bin/env python3
"""
read_flashmatch_hdf5.py

Read HDF5 files created by FlashMatchHDF5Output and prepare data for PyTorch training.
This script demonstrates how to:
1. Read the nested vector data from HDF5
2. Convert to PyTorch tensors
3. Create a custom Dataset for training
4. Set up a DataLoader for batch processing
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path


class FlashMatchHDF5Reader:
    """Read and parse FlashMatch HDF5 files"""
    
    def __init__(self, filepath: str):
        """
        Initialize reader with HDF5 file
        
        Args:
            filepath: Path to HDF5 file created by FlashMatchHDF5Output
        """
        self.filepath = filepath
        self.h5file = None
        self._open_file()
        
    def _open_file(self):
        """Open the HDF5 file for reading"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"HDF5 file not found: {self.filepath}")
        self.h5file = h5py.File(self.filepath, 'r')
        
    def close(self):
        """Close the HDF5 file"""
        if self.h5file:
            self.h5file.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def get_num_entries(self) -> int:
        """Get total number of entries in the file"""
        voxel_group = self.h5file['voxel_data']
        # Count entries by looking for entry_N groups
        count = 0
        while f'entry_{count}' in voxel_group:
            count += 1
        return count
    
    def read_entry(self, entry_idx: int) -> Dict:
        """
        Read a single entry from the HDF5 file
        
        Args:
            entry_idx: Index of the entry to read
            
        Returns:
            Dictionary containing all data for this entry
        """
        entry_name = f'entry_{entry_idx}'
        voxel_group = self.h5file['voxel_data']
        
        if entry_name not in voxel_group:
            raise IndexError(f"Entry {entry_idx} not found in HDF5 file")
            
        entry_group = voxel_group[entry_name]
        
        # Read the voxel data
        data = {
            'planecharge': np.array(entry_group['planecharge']),
            'indices': np.array(entry_group['indices']),
            'avepos': np.array(entry_group['avepos']),
            'centers': np.array(entry_group['centers']),
            # Read event info from attributes
            'run': entry_group.attrs['run'],
            'subrun': entry_group.attrs['subrun'],
            'event': entry_group.attrs['event'],
            'match_index': entry_group.attrs['match_index']
        }
        
        # Read optical flash data if available
        if 'observed_pe_per_pmt' in entry_group:
            data['observed_pe_per_pmt'] = np.array(entry_group['observed_pe_per_pmt'])
        else:
            data['observed_pe_per_pmt'] = np.zeros(32, dtype=np.float32)
            
        if 'predicted_pe_per_pmt' in entry_group:
            data['predicted_pe_per_pmt'] = np.array(entry_group['predicted_pe_per_pmt'])
        else:
            data['predicted_pe_per_pmt'] = np.zeros(32, dtype=np.float32)
            
        if 'observed_total_pe' in entry_group:
            data['observed_total_pe'] = float(entry_group['observed_total_pe'][()])
        else:
            data['observed_total_pe'] = 0.0
            
        if 'predicted_total_pe' in entry_group:
            data['predicted_total_pe'] = float(entry_group['predicted_total_pe'][()])
        else:
            data['predicted_total_pe'] = 0.0
            
        if 'match_type' in entry_group:
            data['match_type'] = int(entry_group['match_type'][()])
        else:
            data['match_type'] = -1
        
        return data
    
    def read_all_entries(self) -> List[Dict]:
        """Read all entries from the file"""
        num_entries = self.get_num_entries()
        entries = []
        for i in range(num_entries):
            entries.append(self.read_entry(i))
        return entries
    
    def print_file_info(self):
        """Print information about the HDF5 file structure"""
        print(f"HDF5 File: {self.filepath}")
        print(f"Total entries: {self.get_num_entries()}")
        
        # Print structure
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                # Print attributes
                for attr_name, attr_val in obj.attrs.items():
                    print(f"{indent}  @{attr_name}: {attr_val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
                
        print("\nFile structure:")
        self.h5file.visititems(print_structure)


class FlashMatchVoxelDataset(Dataset):
    """
    PyTorch Dataset for FlashMatch voxel data
    
    This dataset prepares voxel data for neural network training.
    It can handle variable-length voxel data by padding or truncating.
    """
    
    def __init__(self, 
                 hdf5_files: List[str],
                 max_voxels: int = 1000,
                 transform: Optional[callable] = None,
                 load_to_memory: bool = True):
        """
        Initialize dataset
        
        Args:
            hdf5_files: List of HDF5 file paths
            max_voxels: Maximum number of voxels (for padding/truncation)
            transform: Optional transform to apply to data
            load_to_memory: If True, load all data to memory (faster but uses more RAM)
        """
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.max_voxels = max_voxels
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        # Store file indices and entry counts
        self.file_entries = []  # List of (file_idx, entry_idx) tuples
        self.data_cache = []
        
        # Build index
        self._build_index()
        
        # Optionally load all data to memory
        if self.load_to_memory:
            self._load_all_data()
            
    def _build_index(self):
        """Build an index of all entries across all files"""
        for file_idx, filepath in enumerate(self.hdf5_files):
            with FlashMatchHDF5Reader(filepath) as reader:
                num_entries = reader.get_num_entries()
                for entry_idx in range(num_entries):
                    self.file_entries.append((file_idx, entry_idx))
                    
        print(f"Total dataset size: {len(self.file_entries)} entries from {len(self.hdf5_files)} files")
        
    def _load_all_data(self):
        """Load all data into memory"""
        print("Loading all data to memory...")
        for file_idx, filepath in enumerate(self.hdf5_files):
            with FlashMatchHDF5Reader(filepath) as reader:
                entries = reader.read_all_entries()
                for entry in entries:
                    self.data_cache.append(self._process_entry(entry))
        print(f"Loaded {len(self.data_cache)} entries to memory")
        
    def _process_entry(self, entry: Dict) -> Dict[str, torch.Tensor]:
        """
        Process a single entry and convert to PyTorch tensors
        
        Args:
            entry: Dictionary from HDF5 reader
            
        Returns:
            Dictionary of PyTorch tensors
        """
        # Extract voxel data
        planecharge = entry['planecharge']  # Shape: (n_voxels, 3)
        indices = entry['indices']          # Shape: (n_voxels, 3)
        avepos = entry['avepos']           # Shape: (n_voxels, 3)
        centers = entry['centers']         # Shape: (n_voxels, 3)
        
        n_voxels = len(planecharge)
        
        # Handle variable length by padding or truncating
        if n_voxels > self.max_voxels:
            # Truncate
            planecharge = planecharge[:self.max_voxels]
            indices = indices[:self.max_voxels]
            avepos = avepos[:self.max_voxels]
            centers = centers[:self.max_voxels]
            mask = np.ones(self.max_voxels, dtype=np.float32)
        else:
            # Pad with zeros
            pad_size = self.max_voxels - n_voxels
            if pad_size > 0:
                planecharge = np.pad(planecharge, ((0, pad_size), (0, 0)), mode='constant')
                indices = np.pad(indices, ((0, pad_size), (0, 0)), mode='constant')
                avepos = np.pad(avepos, ((0, pad_size), (0, 0)), mode='constant')
                centers = np.pad(centers, ((0, pad_size), (0, 0)), mode='constant')
            
            # Create mask for valid voxels
            mask = np.zeros(self.max_voxels, dtype=np.float32)
            mask[:n_voxels] = 1.0
        
        # Convert to PyTorch tensors
        processed = {
            'planecharge': torch.from_numpy(planecharge.astype(np.float32)),
            'indices': torch.from_numpy(indices.astype(np.int64)),
            'avepos': torch.from_numpy(avepos.astype(np.float32)),
            'centers': torch.from_numpy(centers.astype(np.float32)),
            'mask': torch.from_numpy(mask),
            'n_voxels': torch.tensor(n_voxels, dtype=torch.int64),
            # Optical flash data
            'observed_pe_per_pmt': torch.from_numpy(entry.get('observed_pe_per_pmt', np.zeros(32)).astype(np.float32)),
            'predicted_pe_per_pmt': torch.from_numpy(entry.get('predicted_pe_per_pmt', np.zeros(32)).astype(np.float32)),
            'observed_total_pe': torch.tensor(entry.get('observed_total_pe', 0.0), dtype=torch.float32),
            'predicted_total_pe': torch.tensor(entry.get('predicted_total_pe', 0.0), dtype=torch.float32),
            'match_type': torch.tensor(entry.get('match_type', -1), dtype=torch.int64),
            # Metadata
            'run': torch.tensor(entry['run'], dtype=torch.int64),
            'subrun': torch.tensor(entry['subrun'], dtype=torch.int64),
            'event': torch.tensor(entry['event'], dtype=torch.int64),
            'match_index': torch.tensor(entry['match_index'], dtype=torch.int64),
        }
        
        # Create combined feature vector (example)
        # You can customize this based on your network architecture
        # Here we concatenate planecharge and normalized position
        features = torch.cat([
            processed['planecharge'],  # Shape: (max_voxels, 3)
            processed['avepos'] / 1000.0,  # Normalize positions (assuming cm scale)
        ], dim=1)  # Shape: (max_voxels, 6)
        
        processed['features'] = features
        
        return processed
        
    def __len__(self):
        return len(self.file_entries)
        
    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        if self.load_to_memory:
            data = self.data_cache[idx]
        else:
            # Load from file on demand
            file_idx, entry_idx = self.file_entries[idx]
            filepath = self.hdf5_files[file_idx]
            with FlashMatchHDF5Reader(filepath) as reader:
                entry = reader.read_entry(entry_idx)
                data = self._process_entry(entry)
                
        if self.transform:
            data = self.transform(data)
            
        return data


class SparseVoxelDataset(Dataset):
    """
    Alternative dataset for sparse 3D convolution networks (MinkowskiEngine)
    This keeps voxel data in sparse format
    """
    
    def __init__(self, hdf5_files: List[str], voxel_size: float = 5.0):
        """
        Initialize sparse dataset
        
        Args:
            hdf5_files: List of HDF5 file paths
            voxel_size: Size of voxels in cm for discretization
        """
        self.hdf5_files = hdf5_files if isinstance(hdf5_files, list) else [hdf5_files]
        self.voxel_size = voxel_size
        self.entries = []
        
        # Load all entries
        for filepath in self.hdf5_files:
            with FlashMatchHDF5Reader(filepath) as reader:
                self.entries.extend(reader.read_all_entries())
                
    def __len__(self):
        return len(self.entries)
        
    def __getitem__(self, idx):
        """Return sparse coordinates and features for MinkowskiEngine"""
        entry = self.entries[idx]
        
        # Get voxel positions and features
        positions = entry['avepos']  # (n_voxels, 3)
        planecharge = entry['planecharge']  # (n_voxels, 3)
        
        # Discretize positions to voxel coordinates
        coords = np.floor(positions / self.voxel_size).astype(np.int32)
        
        # Add batch dimension (required for MinkowskiEngine)
        batch_coords = np.column_stack([
            np.zeros(len(coords), dtype=np.int32),  # batch index
            coords
        ])
        
        # Prepare features
        features = planecharge.astype(np.float32)
        
        return {
            'coordinates': torch.from_numpy(batch_coords),
            'features': torch.from_numpy(features),
            'observed_pe_per_pmt': torch.from_numpy(entry.get('observed_pe_per_pmt', np.zeros(32)).astype(np.float32)),
            'predicted_pe_per_pmt': torch.from_numpy(entry.get('predicted_pe_per_pmt', np.zeros(32)).astype(np.float32)),
            'observed_total_pe': torch.tensor(entry.get('observed_total_pe', 0.0), dtype=torch.float32),
            'predicted_total_pe': torch.tensor(entry.get('predicted_total_pe', 0.0), dtype=torch.float32),
            'match_type': torch.tensor(entry.get('match_type', -1), dtype=torch.int64),
            'metadata': {
                'run': entry['run'],
                'subrun': entry['subrun'],
                'event': entry['event'],
                'match_index': entry['match_index']
            }
        }


def collate_sparse_batch(batch):
    """Custom collate function for sparse voxel data"""
    coordinates = []
    features = []
    observed_pe_per_pmt = []
    predicted_pe_per_pmt = []
    observed_total_pe = []
    predicted_total_pe = []
    match_type = []
    metadata = []
    
    for i, item in enumerate(batch):
        coords = item['coordinates'].clone()
        coords[:, 0] = i  # Update batch index
        coordinates.append(coords)
        features.append(item['features'])
        observed_pe_per_pmt.append(item['observed_pe_per_pmt'])
        predicted_pe_per_pmt.append(item['predicted_pe_per_pmt'])
        observed_total_pe.append(item['observed_total_pe'])
        predicted_total_pe.append(item['predicted_total_pe'])
        match_type.append(item['match_type'])
        metadata.append(item['metadata'])
        
    return {
        'coordinates': torch.cat(coordinates, dim=0),
        'features': torch.cat(features, dim=0),
        'observed_pe_per_pmt': torch.stack(observed_pe_per_pmt),
        'predicted_pe_per_pmt': torch.stack(predicted_pe_per_pmt),
        'observed_total_pe': torch.stack(observed_total_pe),
        'predicted_total_pe': torch.stack(predicted_total_pe),
        'match_type': torch.stack(match_type),
        'metadata': metadata
    }


def example_training_loop():
    """Example of how to use the dataset in a training loop"""
    
    # Setup dataset and dataloader
    hdf5_files = ['output_voxeldata.h5']  # Your HDF5 files
    dataset = FlashMatchVoxelDataset(
        hdf5_files=hdf5_files,
        max_voxels=1000,
        load_to_memory=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Example: Simple neural network for voxel data
    class SimpleVoxelNet(torch.nn.Module):
        def __init__(self, input_dim=6, hidden_dim=128, output_dim=32):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x, mask):
            # x shape: (batch, max_voxels, features)
            # mask shape: (batch, max_voxels)
            
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            
            # Apply mask and aggregate
            x = x * mask.unsqueeze(-1)  # Zero out padded voxels
            x = torch.sum(x, dim=1)  # Aggregate over voxels
            
            return x
    
    # Initialize model
    model = SimpleVoxelNet(input_dim=6, output_dim=32)  # 32 PMTs
    model = model.cuda() if torch.cuda.is_available() else model
    
    # Training loop
    for epoch in range(10):
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            features = batch['features']  # (batch, max_voxels, 6)
            mask = batch['mask']  # (batch, max_voxels)
            
            if torch.cuda.is_available():
                features = features.cuda()
                mask = mask.cuda()
            
            # Forward pass
            predictions = model(features, mask)
            
            # Get observed PMT values as targets
            target_pmt_values = batch['observed_pe_per_pmt']
            if torch.cuda.is_available():
                target_pmt_values = target_pmt_values.cuda()
            
            # Here you would compute loss against PMT predictions
            # loss = criterion(predictions, target_pmt_values)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, "
                      f"Features shape: {features.shape}, "
                      f"Predictions shape: {predictions.shape}")
                
            # Break after a few batches for demo
            if batch_idx > 5:
                break


def main():
    import sys

    """Main function demonstrating usage"""
    
    if len(sys.argv)<=1:
        RUN_TEST = True
        # Create a dummy HDF5 file for testing (you would use your actual file)
        test_file = 'test_voxeldata.h5'
    else:
        RUN_TEST = False
        test_file = sys.argv[1]

    # Read and inspect HDF5 file
    print("=" * 60)
    print("Reading HDF5 file: ",test_file)
    print("=" * 60)
    

    
    # Check if test file exists, if not provide instructions
    if RUN_TEST and not os.path.exists(test_file):
        print(f"\nTest file '{test_file}' not found.")
        print("Please run the C++ example first to generate the HDF5 file:")
        print("  cd data_prep/build")
        print("  ./example_hdf5_usage output_flashmatch.root test_voxeldata.h5")
        print("\nOr use your actual HDF5 files generated from FlashMatchHDF5Output")
        
        # Create a minimal test file for demonstration
        print("\nCreating a minimal test HDF5 file for demonstration...")
        with h5py.File(test_file, 'w') as f:
            voxel_group = f.create_group('voxel_data')
            event_group = f.create_group('event_info')
            
            # Add sample data
            for i in range(3):
                entry = voxel_group.create_group(f'entry_{i}')
                
                # Create dummy voxel data
                n_voxels = np.random.randint(5, 20)
                entry.create_dataset('planecharge', 
                                    data=np.random.rand(n_voxels, 3).astype(np.float32))
                entry.create_dataset('indices', 
                                    data=np.random.randint(0, 1000, (n_voxels, 3)))
                entry.create_dataset('avepos', 
                                    data=np.random.rand(n_voxels, 3).astype(np.float32) * 200)
                entry.create_dataset('centers', 
                                    data=np.random.rand(n_voxels, 3).astype(np.float32) * 200)
                
                # Add dummy optical flash data
                entry.create_dataset('observed_pe_per_pmt',
                                    data=np.random.rand(32).astype(np.float32) * 100)
                entry.create_dataset('predicted_pe_per_pmt',
                                    data=np.random.rand(32).astype(np.float32) * 100)
                entry.create_dataset('observed_total_pe',
                                    data=np.random.rand(1).astype(np.float32) * 1000)
                entry.create_dataset('predicted_total_pe',
                                    data=np.random.rand(1).astype(np.float32) * 1000)
                entry.create_dataset('match_type',
                                    data=np.random.randint(0, 5, 1))
                
                # Add attributes
                entry.attrs['run'] = 1
                entry.attrs['subrun'] = 1
                entry.attrs['event'] = i + 1
                entry.attrs['match_index'] = 0
    
    # Read the file
    with FlashMatchHDF5Reader(test_file) as reader:
        reader.print_file_info()
        
        # Read first entry
        print("\n" + "=" * 60)
        print("First entry data:")
        print("=" * 60)
        entry = reader.read_entry(0)
        for key, value in entry.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                print(f"  First few values: {value[:3] if len(value) > 3 else value}")
            else:
                print(f"{key}: {value}")
    
    # Example 2: Create PyTorch dataset
    print("\n" + "=" * 60)
    print("Example 2: PyTorch Dataset")
    print("=" * 60)
    
    dataset = FlashMatchVoxelDataset(
        hdf5_files=[test_file],
        max_voxels=100,
        load_to_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print("\nSample data shapes:")
    for key, tensor in sample.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
    
    # Example 3: DataLoader
    print("\n" + "=" * 60)
    print("Example 3: DataLoader batching")
    print("=" * 60)
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    
    print("Batch shapes:")
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
    
    # # Example 4: Training loop demo
    # print("\n" + "=" * 60)
    # print("Example 4: Training loop demo")
    # print("=" * 60)
    # example_training_loop()


if __name__ == "__main__":
    main()