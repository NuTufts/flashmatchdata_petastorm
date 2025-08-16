import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add parent directory to path to import the HDF5 reader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from read_flashmatch_hdf5 import FlashMatchHDF5Reader

# ROOT imports
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    print("Warning: ROOT not available. Will save statistics to numpy file instead.")
    HAS_ROOT = False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate means and variances for flashmatch training data')
    parser.add_argument('-i', '--input-file-list', 
                        type=str, 
                        required=True,
                        help='Text file containing paths to HDF5 files')
    parser.add_argument('-o', '--output', 
                        type=str, 
                        default='data_statistics.root',
                        help='Output ROOT file for statistics and plots')
    parser.add_argument('--max-entries', 
                        type=int, 
                        default=None,
                        help='Maximum number of entries to process (for testing)')
    parser.add_argument('--nbins', 
                        type=int, 
                        default=100,
                        help='Number of bins for histograms')
    return parser.parse_args()

def read_file_list(file_list_path: str) -> List[str]:
    """Read list of HDF5 files from text file"""
    files = []
    with open(file_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle relative or absolute paths
                if not os.path.isabs(line):
                    # If relative, make it relative to the file list directory
                    base_dir = os.path.dirname(file_list_path)
                    line = os.path.join(base_dir, line)
                if os.path.exists(line):
                    files.append(line)
                else:
                    print(f"Warning: File not found: {line}")
    return files

class DataStatisticsCalculator:
    """Calculate statistics for flashmatch training data"""
    
    def __init__(self, nbins: int = 100):
        self.nbins = nbins
        self.reset_statistics()
        
    def reset_statistics(self):
        """Reset all statistics"""
        # For online mean and variance calculation
        self.n_samples = 0
        self.n_voxels_total = 0
        
        # Plane charge statistics (3 components)
        self.planecharge_sum = np.zeros(3, dtype=np.float64)
        self.planecharge_sum_sq = np.zeros(3, dtype=np.float64)
        self.planecharge_min = np.full(3, np.inf)
        self.planecharge_max = np.full(3, -np.inf)
        
        # PMT PE statistics (32 PMTs)
        self.pmt_sum = np.zeros(32, dtype=np.float64)
        self.pmt_sum_sq = np.zeros(32, dtype=np.float64)
        self.pmt_min = np.full(32, np.inf)
        self.pmt_max = np.full(32, -np.inf)
        
        # Total PE statistics
        self.total_pe_sum = 0.0
        self.total_pe_sum_sq = 0.0
        self.total_pe_min = np.inf
        self.total_pe_max = -np.inf
        
        # Position statistics (for normalization)
        self.pos_sum = np.zeros(3, dtype=np.float64)
        self.pos_sum_sq = np.zeros(3, dtype=np.float64)
        self.pos_min = np.full(3, np.inf)
        self.pos_max = np.full(3, -np.inf)
        
        # Store all values for histogram creation
        self.all_planecharge = [[] for _ in range(3)]
        self.all_pmt_pe = [[] for _ in range(32)]
        self.all_total_pe = []
        self.all_positions = [[] for _ in range(3)]
        self.all_n_voxels = []
        
    def update(self, entry: Dict):
        """Update statistics with a new entry"""
        # Extract data
        planecharge = entry['planecharge']  # (n_voxels, 3)
        observed_pe = entry['observed_pe_per_pmt']  # (32,)
        total_pe = entry['observed_total_pe']
        positions = entry['avepos']  # (n_voxels, 3)
        
        n_voxels = len(planecharge)
        self.n_samples += 1
        self.n_voxels_total += n_voxels
        self.all_n_voxels.append(n_voxels)
        
        # Update planecharge statistics
        for i in range(3):
            values = planecharge[:, i]
            self.planecharge_sum[i] += np.sum(values)
            self.planecharge_sum_sq[i] += np.sum(values**2)
            self.planecharge_min[i] = min(self.planecharge_min[i], np.min(values))
            self.planecharge_max[i] = max(self.planecharge_max[i], np.max(values))
            self.all_planecharge[i].extend(values.tolist())
            
        # Update PMT statistics
        for i in range(32):
            value = observed_pe[i]
            self.pmt_sum[i] += value
            self.pmt_sum_sq[i] += value**2
            self.pmt_min[i] = min(self.pmt_min[i], value)
            self.pmt_max[i] = max(self.pmt_max[i], value)
            self.all_pmt_pe[i].append(value)
            
        # Update total PE statistics
        self.total_pe_sum += total_pe
        self.total_pe_sum_sq += total_pe**2
        self.total_pe_min = min(self.total_pe_min, total_pe)
        self.total_pe_max = max(self.total_pe_max, total_pe)
        self.all_total_pe.append(total_pe)
        
        # Update position statistics
        for i in range(3):
            values = positions[:, i]
            self.pos_sum[i] += np.sum(values)
            self.pos_sum_sq[i] += np.sum(values**2)
            self.pos_min[i] = min(self.pos_min[i], np.min(values))
            self.pos_max[i] = max(self.pos_max[i], np.max(values))
            self.all_positions[i].extend(values.tolist())
            
    def compute_statistics(self) -> Dict:
        """Compute final statistics"""
        stats = {}
        
        if self.n_samples == 0:
            print("Warning: No samples processed")
            return stats
            
        # Planecharge statistics (per voxel)
        stats['planecharge_mean'] = self.planecharge_sum / self.n_voxels_total
        stats['planecharge_var'] = (self.planecharge_sum_sq / self.n_voxels_total) - stats['planecharge_mean']**2
        stats['planecharge_std'] = np.sqrt(np.maximum(stats['planecharge_var'], 0))
        stats['planecharge_min'] = self.planecharge_min
        stats['planecharge_max'] = self.planecharge_max
        
        # PMT statistics (per event)
        stats['pmt_mean'] = self.pmt_sum / self.n_samples
        stats['pmt_var'] = (self.pmt_sum_sq / self.n_samples) - stats['pmt_mean']**2
        stats['pmt_std'] = np.sqrt(np.maximum(stats['pmt_var'], 0))
        stats['pmt_min'] = self.pmt_min
        stats['pmt_max'] = self.pmt_max
        
        # Total PE statistics
        stats['total_pe_mean'] = self.total_pe_sum / self.n_samples
        stats['total_pe_var'] = (self.total_pe_sum_sq / self.n_samples) - stats['total_pe_mean']**2
        stats['total_pe_std'] = np.sqrt(max(stats['total_pe_var'], 0))
        stats['total_pe_min'] = self.total_pe_min
        stats['total_pe_max'] = self.total_pe_max
        
        # Position statistics (per voxel)
        stats['pos_mean'] = self.pos_sum / self.n_voxels_total
        stats['pos_var'] = (self.pos_sum_sq / self.n_voxels_total) - stats['pos_mean']**2
        stats['pos_std'] = np.sqrt(np.maximum(stats['pos_var'], 0))
        stats['pos_min'] = self.pos_min
        stats['pos_max'] = self.pos_max
        
        # Number of voxels statistics
        stats['n_voxels_mean'] = np.mean(self.all_n_voxels)
        stats['n_voxels_std'] = np.std(self.all_n_voxels)
        stats['n_voxels_min'] = np.min(self.all_n_voxels)
        stats['n_voxels_max'] = np.max(self.all_n_voxels)
        
        stats['n_samples'] = self.n_samples
        stats['n_voxels_total'] = self.n_voxels_total
        
        return stats
        
    def create_root_histograms(self) -> Dict:
        """Create ROOT histograms for all features"""
        if not HAS_ROOT:
            return {}
            
        histograms = {}
        
        # Planecharge histograms
        for i in range(3):
            name = f"h_planecharge_{i}"
            title = f"Plane Charge Component {i};Charge;Entries"
            h = ROOT.TH1F(name, title, self.nbins, 
                         float(self.planecharge_min[i]), 
                         float(self.planecharge_max[i]))
            for val in self.all_planecharge[i]:
                h.Fill(val)
            histograms[name] = h
            
        # PMT PE histograms
        for i in range(32):
            name = f"h_pmt_{i}"
            title = f"PMT {i} PE;Photoelectrons;Entries"
            h = ROOT.TH1F(name, title, self.nbins,
                         float(self.pmt_min[i]),
                         float(max(self.pmt_max[i], self.pmt_min[i] + 1)))
            for val in self.all_pmt_pe[i]:
                h.Fill(val)
            histograms[name] = h
            
        # Combined PMT histogram
        all_pmt_values = []
        for pmt_values in self.all_pmt_pe:
            all_pmt_values.extend(pmt_values)
        h_all_pmt = ROOT.TH1F("h_all_pmt", "All PMT PE;Photoelectrons;Entries",
                              self.nbins, float(np.min(self.pmt_min)), 
                              float(np.max(self.pmt_max)))
        for val in all_pmt_values:
            h_all_pmt.Fill(val)
        histograms["h_all_pmt"] = h_all_pmt
        
        # Total PE histogram
        h_total = ROOT.TH1F("h_total_pe", "Total PE;Total Photoelectrons;Entries",
                           self.nbins, float(self.total_pe_min), 
                           float(self.total_pe_max))
        for val in self.all_total_pe:
            h_total.Fill(val)
        histograms["h_total_pe"] = h_total
        
        # Position histograms
        pos_labels = ['X', 'Y', 'Z']
        for i in range(3):
            name = f"h_position_{pos_labels[i].lower()}"
            title = f"Voxel Position {pos_labels[i]};Position [cm];Entries"
            h = ROOT.TH1F(name, title, self.nbins,
                         float(self.pos_min[i]), 
                         float(self.pos_max[i]))
            for val in self.all_positions[i]:
                h.Fill(val)
            histograms[name] = h

        # 2D position histograms
        # XZ
        h_zx = ROOT.TH2D("h_zx",";z (cm);x (cm)",50,-50,1100,50, -50,300)
        h_xy = ROOT.TH2D("h_xy",";x (cm);y (cm)",50,-50, 300,50,-130,130)
        h_zy = ROOT.TH2D("h_zy",";z (cm);y (cm)",50,-50,1100,50,-130,130)
        for (xval,yval,zval) in zip(self.all_positions[0],self.all_positions[1],self.all_positions[2]):
             h_zx.Fill(zval,xval)
             h_xy.Fill(xval,yval)
             h_zy.Fill(zval,yval)
        histograms['h_zx'] = h_zx
        histograms['h_xy'] = h_xy
        histograms['h_zy'] = h_zy

            
        # Number of voxels histogram
        h_nvoxels = ROOT.TH1F("h_n_voxels", "Number of Voxels per Entry;N Voxels;Entries",
                              50, float(np.min(self.all_n_voxels)), 
                              float(np.max(self.all_n_voxels)))
        for val in self.all_n_voxels:
            h_nvoxels.Fill(val)
        histograms["h_n_voxels"] = h_nvoxels
        
        return histograms

def save_statistics(stats: Dict, histograms: Dict, output_path: str):
    """Save statistics and histograms to ROOT file"""
    if HAS_ROOT and histograms:
        # Save to ROOT file
        root_file = ROOT.TFile(output_path, "RECREATE")
        
        # Create tree for statistics
        tree = ROOT.TTree("statistics", "Data Statistics")
        
        # Create branches for statistics
        stat_values = {}
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                for i, v in enumerate(value):
                    branch_name = f"{key}_{i}"
                    stat_values[branch_name] = np.array([float(v)], dtype=np.float32)
                    tree.Branch(branch_name, stat_values[branch_name], f"{branch_name}/F")
            else:
                stat_values[key] = np.array([float(value)], dtype=np.float32)
                tree.Branch(key, stat_values[key], f"{key}/F")
        
        # Fill the tree
        tree.Fill()
        tree.Write()
        
        # Write histograms
        for hist in histograms.values():
            hist.Write()
            
        root_file.Close()
        print(f"Statistics and histograms saved to {output_path}")
    else:
        # Save to numpy file if ROOT not available
        np_output = output_path.replace('.root', '.npz')
        np.savez(np_output, **stats)
        print(f"Statistics saved to {np_output}")
        
def print_statistics(stats: Dict):
    """Print statistics summary"""
    print("\n" + "="*60)
    print("DATA STATISTICS SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples processed: {stats['n_samples']}")
    print(f"Total voxels processed: {stats['n_voxels_total']}")
    print(f"Average voxels per entry: {stats['n_voxels_mean']:.1f} ± {stats['n_voxels_std']:.1f}")
    print(f"Voxels range: [{stats['n_voxels_min']}, {stats['n_voxels_max']}]")
    
    print("\nPlane Charge Statistics (per voxel):")
    for i in range(3):
        print(f"  Component {i}:")
        print(f"    Mean: {stats['planecharge_mean'][i]:.4f}")
        print(f"    Std:  {stats['planecharge_std'][i]:.4f}")
        print(f"    Range: [{stats['planecharge_min'][i]:.4f}, {stats['planecharge_max'][i]:.4f}]")
    
    print("\nPMT PE Statistics (per event):")
    print(f"  Mean across PMTs: {np.mean(stats['pmt_mean']):.2f}")
    print(f"  Std across PMTs:  {np.mean(stats['pmt_std']):.2f}")
    print(f"  Global range: [{np.min(stats['pmt_min']):.2f}, {np.max(stats['pmt_max']):.2f}]")
    
    print(f"\nTotal PE Statistics:")
    print(f"  Mean: {stats['total_pe_mean']:.2f}")
    print(f"  Std:  {stats['total_pe_std']:.2f}")
    print(f"  Range: [{stats['total_pe_min']:.2f}, {stats['total_pe_max']:.2f}]")
    
    print("\nPosition Statistics (cm):")
    pos_labels = ['X', 'Y', 'Z']
    for i in range(3):
        print(f"  {pos_labels[i]}:")
        print(f"    Mean: {stats['pos_mean'][i]:.2f}")
        print(f"    Std:  {stats['pos_std'][i]:.2f}")
        print(f"    Range: [{stats['pos_min'][i]:.2f}, {stats['pos_max'][i]:.2f}]")
    
    print("\nNormalization Recommendations:")
    print("  For plane charge: subtract mean and divide by std")
    print("  For positions: divide by 100.0 (or use range normalization)")
    print("  For PMT PE: consider log(1 + pe) transformation or divide by total_pe")

def main():
    args = parse_arguments()
    
    # Read file list
    hdf5_files = read_file_list(args.input_file_list)
    if not hdf5_files:
        print("Error: No valid HDF5 files found in file list")
        sys.exit(1)
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Initialize statistics calculator
    calculator = DataStatisticsCalculator(nbins=args.nbins)
    
    # Process files
    total_entries_processed = 0
    for file_path in tqdm(hdf5_files, desc="Processing files"):
        try:
            with FlashMatchHDF5Reader(file_path) as reader:
                num_entries = reader.get_num_entries()
                
                for entry_idx in range(num_entries):
                    if args.max_entries and total_entries_processed >= args.max_entries:
                        break
                        
                    entry = reader.read_entry(entry_idx)
                    calculator.update(entry)
                    total_entries_processed += 1
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
            
        if args.max_entries and total_entries_processed >= args.max_entries:
            print(f"Reached maximum entries limit ({args.max_entries})")
            break
    
    # Compute statistics
    stats = calculator.compute_statistics()
    
    # Print summary
    print_statistics(stats)
    
    # Create histograms
    histograms = calculator.create_root_histograms()
    
    # Save results
    save_statistics(stats, histograms, args.output)
    
    print("\nDone!")

if __name__ == "__main__":
    main()