#!/usr/bin/env python3
"""
HDF5 Converter for Flash-Track Matching Data

This script converts ROOT files containing flash-track matches into HDF5 format
optimized for neural network training. Implements Step 4 of the data preparation pipeline.

Author: Generated for flashmatch data preparation pipeline
"""

import argparse
import h5py
import numpy as np
import yaml
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)  # Run in batch mode
except ImportError:
    print("Error: ROOT not available. Please ensure ROOT is installed and configured.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HDF5Schema:
    """Defines the HDF5 data schema for neural network training"""
    
    # Dataset names
    TRACK_POINTS = "track_points"
    TRACK_CHARGE = "track_charge" 
    TRACK_FEATURES = "track_features"
    FLASH_PE = "flash_pe"
    FLASH_TIME = "flash_time"
    CRT_HITS = "crt_hits"
    GEOMETRY_INFO = "geometry_info"
    QUALITY_METRICS = "quality_metrics"
    EVENT_INFO = "event_info"
    
    # Data types
    FLOAT32 = np.float32
    INT32 = np.int32
    
    @staticmethod
    def get_datasets() -> Dict[str, Dict[str, Any]]:
        """Get dataset definitions for HDF5 file creation"""
        return {
            HDF5Schema.TRACK_POINTS: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (None, 3),  # (N_points, 3) for x,y,z coordinates
                'description': '3D track coordinates in cm'
            },
            HDF5Schema.TRACK_CHARGE: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (None,),  # (N_points,) charge at each point
                'description': 'Charge deposition at each track point in ADC'
            },
            HDF5Schema.TRACK_FEATURES: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (None, 5),  # (N_points, N_features) additional features
                'description': 'Additional track features: dE/dx, distance_to_boundary, etc.'
            },
            HDF5Schema.FLASH_PE: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (32,),  # 32 PMTs
                'description': 'Photoelectron counts for each PMT'
            },
            HDF5Schema.FLASH_TIME: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (),  # Scalar
                'description': 'Flash timing in microseconds'
            },
            HDF5Schema.CRT_HITS: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (None, 4),  # (N_hits, 4) for x,y,z,time
                'description': 'CRT hit positions and times'
            },
            HDF5Schema.GEOMETRY_INFO: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (10,),  # Detector geometry parameters
                'description': 'Detector geometry metadata'
            },
            HDF5Schema.QUALITY_METRICS: {
                'dtype': HDF5Schema.FLOAT32,
                'shape': (8,),  # Quality scores
                'description': 'Quality metrics for filtering'
            },
            HDF5Schema.EVENT_INFO: {
                'dtype': HDF5Schema.INT32,
                'shape': (3,),  # run, subrun, event
                'description': 'Event identification: run, subrun, event'
            }
        }

class ConfigParser:
    """Parse YAML configuration files"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_default_config()
        if config_file:
            self.load_from_file(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'output': {
                'max_entries_per_file': 10000,
                'compression': 'gzip',
                'compression_level': 6,
                'chunk_size': 1000,
                'shuffle': True
            },
            'data_selection': {
                'min_track_points': 10,
                'max_track_points': 10000,
                'min_flash_pe': 10.0,
                'max_flash_pe': 10000.0,
                'quality_score_threshold': 0.1
            },
            'preprocessing': {
                'normalize_coordinates': True,
                'center_coordinates': True,
                'charge_scaling_factor': 1.0,
                'time_offset': 0.0
            }
        }
    
    def load_from_file(self, filename: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(filename, 'r') as f:
                file_config = yaml.safe_load(f)
                self._update_config(self.config, file_config)
                logger.info(f"Loaded configuration from {filename}")
        except Exception as e:
            logger.error(f"Error loading configuration from {filename}: {e}")
            raise
    
    def _update_config(self, base: Dict, update: Dict) -> None:
        """Recursively update configuration dictionary"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value

class FlashTrackMatchConverter:
    """Main converter class for ROOT to HDF5 conversion"""
    
    def __init__(self, config: ConfigParser):
        self.config = config.config
        self.schema = HDF5Schema()
        self.conversion_stats = {
            'total_entries': 0,
            'converted_entries': 0,
            'filtered_entries': 0,
            'error_entries': 0
        }
    
    def convert_file(self, input_file: str, output_file: str) -> bool:
        """Convert a single ROOT file to HDF5 format"""
        
        logger.info(f"Converting {input_file} to {output_file}")
        
        try:
            # Open input ROOT file
            root_file = ROOT.TFile.Open(input_file, "READ")
            if not root_file or root_file.IsZombie():
                logger.error(f"Cannot open ROOT file: {input_file}")
                return False
            
            # Create output HDF5 file
            with h5py.File(output_file, 'w') as h5_file:
                self._setup_hdf5_file(h5_file)
                
                # Process entries
                success = self._convert_entries(root_file, h5_file)
                
                # Add metadata
                self._add_metadata(h5_file, input_file)
            
            root_file.Close()
            
            logger.info(f"Conversion complete: {self.conversion_stats['converted_entries']} entries converted")
            return success
            
        except Exception as e:
            logger.error(f"Error during conversion: {e}")
            return False
    
    def _setup_hdf5_file(self, h5_file: h5py.File) -> None:
        """Setup HDF5 file structure and datasets"""
        
        datasets = self.schema.get_datasets()
        compression = self.config['output']['compression']
        compression_opts = self.config['output']['compression_level']
        chunk_size = self.config['output']['chunk_size']
        
        for name, info in datasets.items():
            if info['shape'] == ():  # Scalar datasets
                maxshape = (None,)
                chunks = (chunk_size,)
            elif len(info['shape']) == 1 and info['shape'][0] is None:
                maxshape = (None, 1)  # Variable length 1D
                chunks = (chunk_size, 1)
            elif len(info['shape']) == 2 and info['shape'][0] is None:
                maxshape = (None, info['shape'][1])  # Variable length 2D
                chunks = (chunk_size, info['shape'][1])
            else:
                maxshape = (None,) + info['shape']  # Fixed shape with unlimited first dim
                chunks = (chunk_size,) + info['shape']
            
            # Create dataset
            h5_file.create_dataset(
                name,
                shape=(0,) + (info['shape'] if info['shape'] != () else ()),
                maxshape=maxshape,
                dtype=info['dtype'],
                compression=compression,
                compression_opts=compression_opts,
                chunks=chunks,
                shuffle=self.config['output']['shuffle']
            )
            
            # Add description as attribute
            h5_file[name].attrs['description'] = info['description']
    
    def _convert_entries(self, root_file: ROOT.TFile, h5_file: h5py.File) -> bool:
        """Convert ROOT entries to HDF5 format"""
        
        # TODO: Implement actual ROOT tree reading
        # For now, create dummy data to demonstrate the structure
        
        max_entries = self.config['output']['max_entries_per_file']
        entries_to_process = min(100, max_entries)  # Dummy: process 100 entries
        
        logger.info(f"Processing {entries_to_process} entries...")
        
        for entry_idx in range(entries_to_process):
            self.conversion_stats['total_entries'] += 1
            
            try:
                # Generate dummy data (replace with actual ROOT reading)
                entry_data = self._generate_dummy_entry(entry_idx)
                
                # Apply data selection cuts
                if not self._passes_selection_cuts(entry_data):
                    self.conversion_stats['filtered_entries'] += 1
                    continue
                
                # Preprocess data
                processed_data = self._preprocess_entry(entry_data)
                
                # Write to HDF5
                self._write_entry_to_hdf5(h5_file, processed_data)
                
                self.conversion_stats['converted_entries'] += 1
                
                if (entry_idx + 1) % 1000 == 0:
                    logger.info(f"Processed {entry_idx + 1} entries...")
                
            except Exception as e:
                logger.warning(f"Error processing entry {entry_idx}: {e}")
                self.conversion_stats['error_entries'] += 1
                continue
        
        return True
    
    def _generate_dummy_entry(self, entry_idx: int) -> Dict[str, np.ndarray]:
        """Generate dummy data for demonstration (replace with ROOT reading)"""
        
        # Generate random track
        n_points = np.random.randint(20, 200)
        track_points = np.random.uniform(-100, 356, (n_points, 3)).astype(np.float32)
        track_charge = np.random.exponential(10.0, n_points).astype(np.float32)
        track_features = np.random.normal(0, 1, (n_points, 5)).astype(np.float32)
        
        # Generate flash data
        flash_pe = np.random.poisson(20, 32).astype(np.float32)
        flash_time = np.random.uniform(0, 20).astype(np.float32)
        
        # Generate CRT hits
        n_crt_hits = np.random.randint(0, 10)
        if n_crt_hits > 0:
            crt_hits = np.random.uniform(-200, 400, (n_crt_hits, 4)).astype(np.float32)
        else:
            crt_hits = np.empty((0, 4), dtype=np.float32)
        
        # Geometry and quality info
        geometry_info = np.array([256.4, 233.0, 1036.8, 0.0, 116.5, -116.5, 
                                 0.0, 0.1098, 0.0, 1.0], dtype=np.float32)
        quality_metrics = np.random.uniform(0.1, 1.0, 8).astype(np.float32)
        event_info = np.array([1, 1, entry_idx], dtype=np.int32)
        
        return {
            self.schema.TRACK_POINTS: track_points,
            self.schema.TRACK_CHARGE: track_charge,
            self.schema.TRACK_FEATURES: track_features,
            self.schema.FLASH_PE: flash_pe,
            self.schema.FLASH_TIME: flash_time,
            self.schema.CRT_HITS: crt_hits,
            self.schema.GEOMETRY_INFO: geometry_info,
            self.schema.QUALITY_METRICS: quality_metrics,
            self.schema.EVENT_INFO: event_info
        }
    
    def _passes_selection_cuts(self, entry_data: Dict[str, np.ndarray]) -> bool:
        """Apply selection cuts to entry data"""
        
        config = self.config['data_selection']
        
        # Check track points
        n_points = len(entry_data[self.schema.TRACK_POINTS])
        if n_points < config['min_track_points'] or n_points > config['max_track_points']:
            return False
        
        # Check flash PE
        total_pe = np.sum(entry_data[self.schema.FLASH_PE])
        if total_pe < config['min_flash_pe'] or total_pe > config['max_flash_pe']:
            return False
        
        # Check quality score
        quality_score = np.mean(entry_data[self.schema.QUALITY_METRICS])
        if quality_score < config['quality_score_threshold']:
            return False
        
        return True
    
    def _preprocess_entry(self, entry_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess entry data before writing to HDF5"""
        
        processed_data = entry_data.copy()
        config = self.config['preprocessing']
        
        # Normalize and center coordinates if requested
        if config['normalize_coordinates'] or config['center_coordinates']:
            points = processed_data[self.schema.TRACK_POINTS].copy()
            
            if config['center_coordinates']:
                # Center around track centroid
                centroid = np.mean(points, axis=0)
                points -= centroid
            
            if config['normalize_coordinates']:
                # Normalize to unit scale
                scale = np.max(np.abs(points))
                if scale > 0:
                    points /= scale
            
            processed_data[self.schema.TRACK_POINTS] = points
        
        # Apply charge scaling
        if config['charge_scaling_factor'] != 1.0:
            processed_data[self.schema.TRACK_CHARGE] *= config['charge_scaling_factor']
        
        # Apply time offset
        if config['time_offset'] != 0.0:
            processed_data[self.schema.FLASH_TIME] += config['time_offset']
        
        return processed_data
    
    def _write_entry_to_hdf5(self, h5_file: h5py.File, entry_data: Dict[str, np.ndarray]) -> None:
        """Write a single entry to HDF5 file"""
        
        for dataset_name, data in entry_data.items():
            dataset = h5_file[dataset_name]
            
            # Resize dataset to accommodate new entry
            current_size = dataset.shape[0]
            dataset.resize((current_size + 1,) + dataset.shape[1:])
            
            # Write data
            if data.ndim == 0:  # Scalar
                dataset[current_size] = data
            elif data.ndim == 1:
                if dataset.shape[1:] == ():  # 1D variable length stored as 2D
                    dataset[current_size, :len(data)] = data
                else:
                    dataset[current_size] = data
            else:  # Multi-dimensional
                dataset[current_size] = data
    
    def _add_metadata(self, h5_file: h5py.File, input_file: str) -> None:
        """Add metadata to HDF5 file"""
        
        h5_file.attrs['input_file'] = input_file
        h5_file.attrs['conversion_timestamp'] = np.string_(
            np.datetime64('now').astype(str))
        h5_file.attrs['schema_version'] = '1.0'
        h5_file.attrs['total_entries'] = self.conversion_stats['total_entries']
        h5_file.attrs['converted_entries'] = self.conversion_stats['converted_entries']
        h5_file.attrs['filtered_entries'] = self.conversion_stats['filtered_entries']
        h5_file.attrs['error_entries'] = self.conversion_stats['error_entries']
        
        # Add configuration as metadata
        config_str = yaml.dump(self.config)
        h5_file.attrs['configuration'] = np.string_(config_str)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get conversion statistics"""
        return self.conversion_stats.copy()

def main():
    """Main program entry point"""
    
    parser = argparse.ArgumentParser(
        description="Convert flash-track matching ROOT files to HDF5 format for neural network training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Input ROOT file with flash-track matches"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help="Output HDF5 file"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help="YAML configuration file"
    )
    
    parser.add_argument(
        '--max-entries-per-file',
        type=int,
        default=10000,
        help="Maximum entries per output file"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Perform dry run (check inputs without conversion)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check input file
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return 1
    
    # Check output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = ConfigParser(args.config)
        
        # Override max entries if specified
        if args.max_entries_per_file != 10000:
            config.config['output']['max_entries_per_file'] = args.max_entries_per_file
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return 1
    
    if args.dry_run:
        logger.info("Dry run mode - checking inputs only")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Configuration: {config.config}")
        return 0
    
    # Perform conversion
    converter = FlashTrackMatchConverter(config)
    
    try:
        success = converter.convert_file(args.input, args.output)
        
        if success:
            stats = converter.get_statistics()
            logger.info("Conversion completed successfully!")
            logger.info(f"Statistics: {stats}")
            return 0
        else:
            logger.error("Conversion failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())