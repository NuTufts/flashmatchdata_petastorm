import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from math import log
from array import array

# Add parent directory to path to import the HDF5 reader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from read_flashmatch_hdf5 import FlashMatchHDF5Reader

# ROOT imports
import ROOT as rt

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate means and variances for flashmatch training data')
    parser.add_argument('-i', '--input-file-list', 
                        type=str, 
                        required=True,
                        help='Text file containing paths to HDF5 files')
    parser.add_argument('-o', '--output', 
                        type=str, 
                        default='dataset_ly_analysis.root',
                        help='Output ROOT file for statistics and plots')
    parser.add_argument('--max-entries', 
                        type=int, 
                        default=None,
                        help='Maximum number of entries to process (for testing)')
    # parser.add_argument('--nbins', 
    #                     type=int, 
    #                     default=10000,
    #                     help='Number of bins for histograms')
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


def main():
    args = parse_arguments()
    
    # Read file list
    hdf5_files = read_file_list(args.input_file_list)
    if not hdf5_files:
        print("Error: No valid HDF5 files found in file list")
        sys.exit(1)
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")

    out = rt.TFile( args.output, 'recreate')
    ttree = rt.TTree("lyana", "light yield analysis tree")
    ly = array('f',[0.0])
    petot = array('f',[0.0])
    pemax = array('f',[0.0])
    qsum  = array('f',[0.0])
    qmax  = array('f',[0.0])
    qx    = array('f',[0.0])
    qy    = array('f',[0.0])
    qz    = array('f',[0.0])
    ttree.Branch('ly',ly,'ly/F')
    ttree.Branch('petot',petot,'petot/F')
    ttree.Branch('pemax',pemax,'pemax/F')
    ttree.Branch('qsum',qsum,'qsum/F')
    ttree.Branch('qmax',qmax,'qmax/F')
    ttree.Branch('qx',qx,'qx/F')
    ttree.Branch('qy',qy,'qy/F')
    ttree.Branch('qz',qz,'qz/F')
        
    # Process files
    total_entries_processed = 0
    badlist = []
    for file_path in tqdm(hdf5_files, desc="Processing files"):
        try:
            with FlashMatchHDF5Reader(file_path) as reader:
                num_entries = reader.get_num_entries()
                
                for entry_idx in range(num_entries):
                    if args.max_entries and total_entries_processed >= args.max_entries:
                        break
                        
                    entry = reader.read_entry(entry_idx)
                    planecharge = entry['planecharge']
                    avepos      = entry['avepos'] # (N,3)
                    if planecharge.shape[0]==0:
                        continue
                    qmean = np.mean( planecharge, axis=1 )
                    qsum[0] = np.sum(qmean)
                    qmax[0] = np.max(qmean)
                    pe_per_pmt  = entry['observed_pe_per_pmt']
                    pemax[0]    = np.max(pe_per_pmt)
                    petot[0]    = entry['observed_total_pe']
                    pe_per_pmt_ubmodel = entry['predicted_pe_per_pmt']
                    if (qsum[0]>0.0 ):
                        ly[0] = petot[0]/qsum[0] 
                        qx[0] = np.sum(avepos[:,0]*qmean)/qsum[0]
                        qy[0] = np.sum(avepos[:,1]*qmean)/qsum[0]
                        qz[0] = np.sum(avepos[:,2]*qmean)/qsum[0]
                    else:
                        ly[0] = -1.0

                    # print(type(planecharge))
                    # print(" planecharge=",planecharge.shape,"qmean=",qmean.shape," petot=",petot[0]," qsum=",qsum[0])
                    # print(entry.keys())
                    #calculator.update(entry)
                    total_entries_processed += 1

                    ttree.Fill()
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            badlist.append(file_path)
            continue
            
        if args.max_entries and total_entries_processed >= args.max_entries:
            print(f"Reached maximum entries limit ({args.max_entries})")
            break
    
    
    print("\nDone!")
    out.Write()
    print("Number of files with errors: ",len(badlist))
    out_txt = args.output.replace(".root",".txt")
    badfilepath = f"badlist_{out_txt}"
    fbadfile = open( badfilepath, 'w' )
    for fpath in badlist:
        print(fpath,file=fbadfile)
    fbadfile.close()

if __name__ == "__main__":
    main()
