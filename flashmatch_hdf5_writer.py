#!/usr/bin/env python
import os
import sys
import argparse
import h5py
import numpy as np
from math import fabs

# ROOT/larcv/larlite imports
import ROOT as rt
from larcv import larcv
from larlite import larlite
from ublarcvapp import ublarcvapp
from larflow import larflow
from ROOT import std

rt.gStyle.SetOptStat(0)
rt.gROOT.ProcessLine("gErrorIgnoreLevel = 3002;")

def makeMCFlashMatchData(io_larcv, io_ll, fmutil, voxelizer, 
                         out_dict, adc_name="wiremc",
                         truth_correct_tdrift=True):
    """
    Create MC flashmatch training data from ROOT files.
    
    Args:
        io_larcv: larcv IOManager for truth data
        io_ll: larlite storage_manager for MC info and optical reconstruction
        fmutil: OpModelMCDataPrep utility for flash matching
        voxelizer: VoxelizeTriplets for spacepoint processing
        out_dict: Output dictionary to store data
        adc_name: ADC producer name
        truth_correct_tdrift: Whether to correct for drift time using truth
    """
    
    # Get run/subrun/event info
    run = io_ll.run_id()
    subrun = io_ll.subrun_id()
    event = io_ll.event_id()
    
    # Get the source ROOT filename from larlite
    sourcefile = os.path.basename(io_ll.get_data_file_name(0))
    
    # Process voxelization with truth labels
    voxelizer.process_fullchain_withtruth(io_larcv, io_ll, adc_name, adc_name, truth_correct_tdrift)
    
    # Match reco flashes to true track and shower information
    fmutil.process(io_ll, voxelizer)
    fmutil.printMatches()
    
    # Get TPC boundaries in voxel indices
    tpc_origin = std.vector("float")(3)
    tpc_origin[0] = 0.0
    tpc_origin[1] = -117.0
    tpc_origin[2] = 0.0
    
    tpc_end = std.vector("float")(3)
    tpc_end[0] = 256.0
    tpc_end[1] = 117.0
    tpc_end[2] = 1036.0
    
    index_tpc_origin = [voxelizer.get_axis_voxel(i, tpc_origin[i]) for i in range(3)]
    index_tpc_end = [voxelizer.get_axis_voxel(i, tpc_end[i]) for i in range(3)]
    
    # Initialize lists if not present
    for key in ['sourcefile', 'run', 'subrun', 'event', 'matchindex', 
                'ancestorid', 'coord', 'feat', 'flashpe']:
        if key not in out_dict:
            out_dict[key] = []
            
    naccepted = 0
    
    # Loop through the truth-matched reconstruction optical flashes
    for iflash in range(fmutil.recoflash_v.size()):
        flash = fmutil.recoflash_v.at(iflash)
        
        # Skip null flash outliers
        if flash.producerid == -1:
            continue
            
        # Get data dictionary for this flash
        data_dict = fmutil.make_opmodel_data_dict(flash, voxelizer, io_ll)
        
        # Define TPC window boundaries
        win_xmin = index_tpc_origin[0]
        win_xmax = index_tpc_end[0]
        
        # Crop voxels to TPC boundaries
        voxcoord_above_xbound = (data_dict['voxcoord'][:,0] >= win_xmin) * (data_dict['voxcoord'][:,0] <= win_xmax)
        voxcoord_above_ybound = (data_dict['voxcoord'][:,1] >= index_tpc_origin[1]) * (data_dict['voxcoord'][:,1] <= index_tpc_end[1]+1)
        voxcoord_above_zbound = (data_dict['voxcoord'][:,2] >= index_tpc_origin[2]) * (data_dict['voxcoord'][:,2] <= index_tpc_end[2]+1)
        
        # Create mask for voxels inside TPC
        intpc = voxcoord_above_xbound * voxcoord_above_ybound * voxcoord_above_zbound
        
        # Select voxels inside TPC
        voxcoord_intpc = data_dict['voxcoord'][intpc[:], :]
        voxfeat_intpc = data_dict['voxcharge'][intpc[:], :]
        
        # Subtract off the lower-x TPC boundary index
        voxcoord_intpc[:, 0] -= win_xmin
        
        # Skip if no voxels remain
        if voxcoord_intpc.shape[0] < 1:
            continue
            
        # Store the data
        out_dict['sourcefile'].append(sourcefile)
        out_dict['run'].append(run)
        out_dict['subrun'].append(subrun)
        out_dict['event'].append(event)
        out_dict['matchindex'].append(naccepted)
        out_dict['ancestorid'].append(int(flash.ancestorid))
        out_dict['coord'].append(voxcoord_intpc)
        out_dict['feat'].append(voxfeat_intpc)
        out_dict['flashpe'].append(data_dict['flashpe'])
        
        naccepted += 1
        
    return naccepted


def main():
    parser = argparse.ArgumentParser(description='Create FlashMatch HDF5 training data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('-lcv', '--in-larcvtruth', type=str, required=True,
                        help='Path to larcv truth root file')
    parser.add_argument('-mc', '--in-mcinfo', type=str, required=True,
                        help='Path to mcinfo root file')
    parser.add_argument('-op', '--in-opreco', type=str, required=True,
                        help='Path to opreco root file')
    parser.add_argument('-v', '--verbosity', type=int, default=0,
                        help='Set Verbosity Level [0=quiet, 2=debug]')
    parser.add_argument('-e', '--entry', type=int, default=None,
                        help='Run specific entry')
    parser.add_argument('-n', '--num-entries', type=int, default=None,
                        help='Run n entries')
    
    args = parser.parse_args()
    
    # Setup ADC name
    adc_name = "wiremc"
    
    # flashmatch class from ublarcvapp
    fmutil = larflow.opticalmodel.OpModelMCDataPrep()
    fmutil.setVerboseLevel(args.verbosity)
    
    # drift velocity
    from ROOT import larutil
    larp = larutil.LArProperties.GetME()
    driftv = larp.DriftVelocity()
    
    # c++ classes that provides spacepoint labels
    voxelizer = larflow.voxelizer.VoxelizeTriplets()
    voxel_len = 5.0
    voxelizer.set_voxel_size_cm(voxel_len)  # re-define voxels to 5 cm spaces
    ndims_v = voxelizer.get_dim_len()
    origin_v = voxelizer.get_origin()
    
    print("VOXELIZER SETUP =====================")
    print("origin: (", origin_v[0], ",", origin_v[1], ",", origin_v[2], ")")
    print("ndims: (", ndims_v[0], ",", ndims_v[1], ",", ndims_v[2], ")")
    
    # Open input files
    iolcv = larcv.IOManager(larcv.IOManager.kREAD, "io", larcv.IOManager.kTickBackward)
    iolcv.add_in_file(args.in_larcvtruth)
    iolcv.specify_data_read(larcv.kProductImage2D, "wire")
    iolcv.specify_data_read(larcv.kProductImage2D, "wiremc")
    iolcv.specify_data_read(larcv.kProductChStatus, "wire")
    iolcv.specify_data_read(larcv.kProductChStatus, "wiremc")
    iolcv.specify_data_read(larcv.kProductImage2D, "ancestor")
    iolcv.specify_data_read(larcv.kProductImage2D, "instance")
    iolcv.specify_data_read(larcv.kProductImage2D, "segment")
    iolcv.specify_data_read(larcv.kProductImage2D, "larflow")
    iolcv.reverse_all_products()
    iolcv.initialize()
    
    ioll = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll.add_in_filename(args.in_opreco)
    ioll.add_in_filename(args.in_mcinfo)
    ioll.set_data_to_read(larlite.data.kMCTrack, "mcreco")
    ioll.set_data_to_read(larlite.data.kMCShower, "mcreco")
    ioll.set_data_to_read(larlite.data.kMCTruth, "generator")
    ioll.set_data_to_read(larlite.data.kOpFlash, "simpleFlashBeam")
    ioll.set_data_to_read(larlite.data.kOpFlash, "simpleFlashCosmic")
    ioll.open()
    
    nentries = iolcv.get_n_entries()
    print("Number of entries (LARCV): ", nentries)
    print("Number of entries (LARLITE): ", ioll.get_entries())
    
    # Determine range of entries to process
    start_entry = 0
    end_entry = nentries
    if args.entry is not None:
        start_entry = args.entry
        end_entry = start_entry + 1
    if args.num_entries is not None:
        if end_entry < 0:
            end_entry = nentries
        else:
            end_entry = start_entry + args.num_entries
            if end_entry > nentries:
                end_entry = nentries
                
    print("RUN FROM ENTRY", start_entry, "to", end_entry)
    
    # Output dictionary
    out_dict = {}
    
    # Process entries
    for ientry in range(start_entry, end_entry):
        print()
        print("==========================")
        print("===[ EVENT", ientry, "]===")
        
        fmutil.clear()
        
        iolcv.read_entry(ientry)
        ioll.go_to(ientry)
        
        naccepted = makeMCFlashMatchData(iolcv, ioll, fmutil, voxelizer, 
                                       out_dict, adc_name=adc_name,
                                       truth_correct_tdrift=True)
        
        print("number of training examples made in this event:", naccepted)
    
    # Close input files
    iolcv.finalize()
    ioll.close()
    
    # Write output HDF5 file
    if len(out_dict.get('coord', [])) == 0:
        print("No data to write, exiting")
        return
        
    output_file = args.output
    print(f"Writing {len(out_dict['coord'])} entries to {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Create datasets for scalar values
        f.create_dataset('sourcefile', data=np.array(out_dict['sourcefile'], dtype='S256'))
        f.create_dataset('run', data=np.array(out_dict['run'], dtype=np.int32))
        f.create_dataset('subrun', data=np.array(out_dict['subrun'], dtype=np.int32))
        f.create_dataset('event', data=np.array(out_dict['event'], dtype=np.int32))
        f.create_dataset('matchindex', data=np.array(out_dict['matchindex'], dtype=np.int32))
        f.create_dataset('ancestorid', data=np.array(out_dict['ancestorid'], dtype=np.int32))
        
        # Variable length datasets for arrays
        coord_dtype = h5py.special_dtype(vlen=np.dtype('int64'))
        feat_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
        flashpe_dtype = h5py.special_dtype(vlen=np.dtype('float32'))
        
        coord_dset = f.create_dataset('coord', (len(out_dict['coord']),), dtype=coord_dtype)
        feat_dset = f.create_dataset('feat', (len(out_dict['feat']),), dtype=feat_dtype)
        flashpe_dset = f.create_dataset('flashpe', (len(out_dict['flashpe']),), dtype=flashpe_dtype)
        
        for i in range(len(out_dict['coord'])):
            coord_dset[i] = out_dict['coord'][i].flatten()
            feat_dset[i] = out_dict['feat'][i].flatten()
            flashpe_dset[i] = out_dict['flashpe'][i].flatten()
            
        # Store metadata
        f.attrs['num_entries'] = len(out_dict['coord'])
        f.attrs['npmts'] = 32
        f.attrs['voxel_size_cm'] = voxel_len
        
    print(f"Successfully wrote {output_file}")


if __name__ == "__main__":
    main()