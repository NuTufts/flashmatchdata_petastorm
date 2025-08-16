import os, sys
import h5py
import numpy as np
import random

def check_all_shapes(filelist_path):
    """Check all HDF5 files for shape issues"""
    
    # Read file list
    files = []
    with open(filelist_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                files.append(line)
    
    print(f"Checking {len(files)} files...")
    
    problem_files = []
    shape_stats = {
        'planecharge_dims': [],
        'indices_dims': [],
        'avepos_dims': [],
        'centers_dims': []
    }
    
    for file_idx, filepath in enumerate(files):
        if file_idx % 100 == 0:
            print(f"Checked {file_idx}/{len(files)} files...")
            
        try:
            with h5py.File(filepath, 'r') as hf:
                if 'voxel_data' not in hf:
                    problem_files.append((filepath, "No voxel_data group"))
                    continue
                    
                voxel_group = hf['voxel_data']
                
                # Check a few entries per file
                for entry_idx in range(min(3, len(voxel_group))):
                    entry_name = f'entry_{entry_idx}'
                    if entry_name not in voxel_group:
                        break
                        
                    entry = voxel_group[entry_name]
                    
                    # Check each dataset
                    for key in ['planecharge', 'indices', 'avepos', 'centers']:
                        if key in entry:
                            data = entry[key]
                            shape = data.shape
                            
                            # Check if second dimension is 3 (for 3D coordinates/features)
                            if len(shape) != 2:
                                problem_files.append((filepath, f"{key} has {len(shape)} dimensions, expected 2"))
                                print(f"ERROR in {filepath}, entry {entry_idx}: {key} shape = {shape}")
                            elif shape[1] != 3:
                                problem_files.append((filepath, f"{key} has shape[1]={shape[1]}, expected 3"))
                                print(f"ERROR in {filepath}, entry {entry_idx}: {key} shape = {shape}")
                            
                            # Sometimes data might be corrupted
                            try:
                                actual_data = np.array(data)
                                if actual_data.shape != shape:
                                    problem_files.append((filepath, f"{key} actual shape differs from reported shape"))
                                    print(f"ERROR in {filepath}: {key} reported shape={shape}, actual shape={actual_data.shape}")
                            except Exception as e:
                                problem_files.append((filepath, f"Cannot read {key}: {e}"))
                                print(f"ERROR reading {filepath} {key}: {e}")
                        
        except Exception as e:
            problem_files.append((filepath, str(e)))
            print(f"Cannot open file {filepath}: {e}")
    
    print(f"\nFound {len(problem_files)} problematic files")
    if problem_files:
        print("\nFirst 10 problematic files:")
        for filepath, error in problem_files[:10]:
            print(f"  {filepath}: {error}")
    
    return problem_files

def test_specific_entry(filepath, entry_idx):
    """Test a specific entry in detail"""
    print(f"\nTesting {filepath}, entry {entry_idx}")
    
    with h5py.File(filepath, 'r') as hf:
        entry = hf[f'voxel_data/entry_{entry_idx}']
        
        for key in ['planecharge', 'indices', 'avepos', 'centers']:
            if key in entry:
                data = np.array(entry[key])
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                print(f"    First row: {data[0] if len(data) > 0 else 'empty'}")
                print(f"    Min: {np.min(data, axis=0) if len(data) > 0 else 'N/A'}")
                print(f"    Max: {np.max(data, axis=0) if len(data) > 0 else 'N/A'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_shapes.py <filelist>")
        sys.exit(1)
    
    filelist = sys.argv[1]
    problem_files = check_all_shapes(filelist)
    
    # If we found problems, test the first problematic file in detail
    if problem_files:
        print("\n" + "="*60)
        print("Testing first problematic file in detail...")
        test_specific_entry(problem_files[0][0], 0)