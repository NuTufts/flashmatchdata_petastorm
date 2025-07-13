#!/usr/bin/env python3
"""
Flash-Track Match Visualization Tool

This script creates visualizations of flash-track matches for quality control
and debugging of the data preparation pipeline.

Usage:
    python plot_flash_matches.py --input matched_data.root --output plots/
    python plot_flash_matches.py --input training_data.h5 --format hdf5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FlashMatchVisualizer:
    """Main visualization class for flash-track matches"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.detector_bounds = {
            'x': (0.0, 256.4),
            'y': (-116.5, 116.5),
            'z': (0.0, 1036.8)
        }
        self.pmt_positions = self._get_pmt_positions()
    
    def _get_pmt_positions(self) -> np.ndarray:
        """Get PMT positions (simplified MicroBooNE geometry)"""
        # Simplified PMT positions - real implementation would load from geometry
        y_positions = np.linspace(-100, 100, 32)
        z_positions = np.full(32, 518.4)  # Middle of detector in Z
        x_positions = np.full(32, 256.4)  # At anode
        
        return np.column_stack([x_positions, y_positions, z_positions])
    
    def plot_track_3d(self, track_points: np.ndarray, track_charge: np.ndarray = None,
                      flash_center: np.ndarray = None, title: str = "3D Track Visualization") -> plt.Figure:
        """Plot 3D track with optional charge information and flash center"""
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot track points
        if track_charge is not None:
            scatter = ax.scatter(track_points[:, 0], track_points[:, 1], track_points[:, 2],
                               c=track_charge, cmap='plasma', s=20, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Charge [ADC]', shrink=0.5)
        else:
            ax.scatter(track_points[:, 0], track_points[:, 1], track_points[:, 2],
                      c='blue', s=20, alpha=0.7)
        
        # Plot track line
        ax.plot(track_points[:, 0], track_points[:, 1], track_points[:, 2],
               'b-', alpha=0.5, linewidth=1)
        
        # Plot flash center if provided
        if flash_center is not None:
            ax.scatter(*flash_center, c='red', s=100, marker='*', 
                      label='Flash Center', alpha=0.8)
        
        # Plot detector boundaries
        self._add_detector_outline_3d(ax)
        
        # Plot PMT positions
        ax.scatter(self.pmt_positions[:, 0], self.pmt_positions[:, 1], self.pmt_positions[:, 2],
                  c='orange', s=30, marker='s', alpha=0.6, label='PMTs')
        
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        ax.set_title(title)
        ax.legend()
        
        return fig
    
    def plot_flash_pe_distribution(self, pe_values: np.ndarray, 
                                  predicted_pe: np.ndarray = None,
                                  title: str = "Flash PE Distribution") -> plt.Figure:
        """Plot PMT PE distribution"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Bar plot of PE values
        pmt_ids = np.arange(len(pe_values))
        bars = ax1.bar(pmt_ids, pe_values, alpha=0.7, label='Observed PE')
        
        if predicted_pe is not None:
            ax1.bar(pmt_ids, predicted_pe, alpha=0.5, label='Predicted PE')
        
        ax1.set_xlabel('PMT ID')
        ax1.set_ylabel('PE Count')
        ax1.set_title('PE by PMT')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2D representation of PMT array
        # Arrange PMTs in approximate geometric layout
        pmt_y = self.pmt_positions[:, 1]
        pmt_z = np.linspace(100, 900, 32)  # Approximate Z positions
        
        scatter = ax2.scatter(pmt_z, pmt_y, c=pe_values, cmap='viridis', 
                            s=100, alpha=0.8)
        plt.colorbar(scatter, ax=ax2, label='PE Count')
        
        ax2.set_xlabel('Z Position [cm]')
        ax2.set_ylabel('Y Position [cm]')
        ax2.set_title('PMT Array Response')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_timing_correlation(self, flash_times: np.ndarray, track_times: np.ndarray,
                               title: str = "Flash-Track Timing Correlation") -> plt.Figure:
        """Plot timing correlation between flashes and tracks"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Scatter plot of timing correlation
        ax1.scatter(track_times, flash_times, alpha=0.6)
        
        # Add diagonal line for perfect correlation
        min_time = min(np.min(track_times), np.min(flash_times))
        max_time = max(np.max(track_times), np.max(flash_times))
        ax1.plot([min_time, max_time], [min_time, max_time], 'r--', alpha=0.5, label='Perfect correlation')
        
        ax1.set_xlabel('Track Time [μs]')
        ax1.set_ylabel('Flash Time [μs]')
        ax1.set_title('Time Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of time differences
        time_diff = flash_times - track_times
        ax2.hist(time_diff, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='Perfect match')
        ax2.axvline(np.mean(time_diff), color='orange', linestyle='-', 
                   label=f'Mean: {np.mean(time_diff):.2f} μs')
        
        ax2.set_xlabel('Time Difference [μs]')
        ax2.set_ylabel('Count')
        ax2.set_title('Time Difference Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_match_quality_metrics(self, match_scores: np.ndarray, 
                                  spatial_distances: np.ndarray,
                                  time_differences: np.ndarray,
                                  title: str = "Match Quality Metrics") -> plt.Figure:
        """Plot various match quality metrics"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Match score distribution
        ax1.hist(match_scores, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(match_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(match_scores):.3f}')
        ax1.set_xlabel('Match Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Match Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spatial distance distribution
        ax2.hist(spatial_distances, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(spatial_distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(spatial_distances):.1f} cm')
        ax2.set_xlabel('Spatial Distance [cm]')
        ax2.set_ylabel('Count')
        ax2.set_title('Spatial Distance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Time difference distribution
        ax3.hist(time_differences, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(time_differences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(time_differences):.2f} μs')
        ax3.set_xlabel('Time Difference [μs]')
        ax3.set_ylabel('Count')
        ax3.set_title('Time Difference Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 2D correlation plot
        scatter = ax4.scatter(spatial_distances, time_differences, c=match_scores, 
                            cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label='Match Score')
        ax4.set_xlabel('Spatial Distance [cm]')
        ax4.set_ylabel('Time Difference [μs]')
        ax4.set_title('Distance vs Time vs Score')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_crt_correlation(self, cosmic_tracks: List[np.ndarray], 
                           crt_hits: List[np.ndarray],
                           title: str = "CRT Correlation") -> plt.Figure:
        """Plot CRT hit correlation with cosmic tracks"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot track and CRT hit positions
        for i, track in enumerate(cosmic_tracks):
            ax1.plot(track[:, 2], track[:, 1], alpha=0.7, 
                    label=f'Track {i}' if i < 5 else "")
        
        for i, hits in enumerate(crt_hits):
            if len(hits) > 0:
                ax1.scatter(hits[:, 2], hits[:, 1], s=50, marker='s', 
                          alpha=0.8, label=f'CRT Hits {i}' if i < 5 else "")
        
        ax1.set_xlabel('Z [cm]')
        ax1.set_ylabel('Y [cm]')
        ax1.set_title('Y-Z Projection')
        if len(cosmic_tracks) <= 5:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot timing correlation
        track_times = []
        crt_times = []
        for i, hits in enumerate(crt_hits):
            if len(hits) > 0 and i < len(cosmic_tracks):
                # Dummy track time calculation
                track_time = i * 5.0  # Simplified
                for hit in hits:
                    track_times.append(track_time)
                    crt_times.append(hit[3] if len(hit) > 3 else 0)
        
        if track_times and crt_times:
            ax2.scatter(track_times, crt_times, alpha=0.6)
            ax2.set_xlabel('Track Time [μs]')
            ax2.set_ylabel('CRT Time [ns]')
            ax2.set_title('Timing Correlation')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No CRT data available', 
                    transform=ax2.transAxes, ha='center', va='center')
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def _add_detector_outline_3d(self, ax):
        """Add detector outline to 3D plot"""
        # Draw detector boundaries
        x_min, x_max = self.detector_bounds['x']
        y_min, y_max = self.detector_bounds['y']
        z_min, z_max = self.detector_bounds['z']
        
        # Draw edges of detector volume
        edges = [
            [(x_min, y_min, z_min), (x_max, y_min, z_min)],
            [(x_min, y_max, z_min), (x_max, y_max, z_min)],
            [(x_min, y_min, z_max), (x_max, y_min, z_max)],
            [(x_min, y_max, z_max), (x_max, y_max, z_max)],
            [(x_min, y_min, z_min), (x_min, y_max, z_min)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min)],
            [(x_min, y_min, z_max), (x_min, y_max, z_max)],
            [(x_max, y_min, z_max), (x_max, y_max, z_max)],
            [(x_min, y_min, z_min), (x_min, y_min, z_max)],
            [(x_max, y_min, z_min), (x_max, y_min, z_max)],
            [(x_min, y_max, z_min), (x_min, y_max, z_max)],
            [(x_max, y_max, z_min), (x_max, y_max, z_max)]
        ]
        
        for edge in edges:
            ax.plot3D(*zip(*edge), 'k-', alpha=0.2, linewidth=0.5)

def generate_dummy_data() -> Dict:
    """Generate dummy data for testing visualization"""
    
    # Generate dummy track
    n_points = 100
    track_points = np.random.uniform([0, -100, 0], [250, 100, 1000], (n_points, 3))
    track_charge = np.random.exponential(10.0, n_points)
    
    # Generate dummy flash
    flash_pe = np.random.poisson(20, 32)
    flash_center = np.array([128, 0, 500])
    flash_time = 5.0
    
    # Generate dummy matches
    n_matches = 50
    match_scores = np.random.beta(2, 5, n_matches)
    spatial_distances = np.random.exponential(30, n_matches)
    time_differences = np.random.normal(0, 1, n_matches)
    
    # Generate dummy CRT data
    crt_hits = [np.random.uniform([-50, -100, 0, 0], [300, 100, 1000, 1000], (5, 4))]
    
    return {
        'track_points': track_points,
        'track_charge': track_charge,
        'flash_pe': flash_pe,
        'flash_center': flash_center,
        'flash_time': flash_time,
        'match_scores': match_scores,
        'spatial_distances': spatial_distances,
        'time_differences': time_differences,
        'crt_hits': crt_hits
    }

def main():
    """Main program entry point"""
    
    parser = argparse.ArgumentParser(
        description="Visualize flash-track matching results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help="Input file (ROOT or HDF5)"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='plots',
        help="Output directory for plots"
    )
    
    parser.add_argument(
        '--format',
        choices=['root', 'hdf5', 'dummy'],
        default='dummy',
        help="Input file format"
    )
    
    parser.add_argument(
        '--event-id',
        type=int,
        default=0,
        help="Event ID to visualize"
    )
    
    parser.add_argument(
        '--entry-range',
        type=str,
        default='0:10',
        help="Entry range to process (start:end)"
    )
    
    parser.add_argument(
        '--plots',
        nargs='+',
        choices=['track_3d', 'flash_pe', 'timing', 'quality', 'crt'],
        default=['track_3d', 'flash_pe', 'quality'],
        help="Types of plots to generate"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize visualizer
    visualizer = FlashMatchVisualizer()
    
    # Load data (for now, use dummy data)
    if args.format == 'dummy' or not args.input:
        print("Using dummy data for demonstration...")
        data = generate_dummy_data()
    else:
        print(f"Loading data from {args.input} (format: {args.format})")
        # TODO: Implement actual data loading
        data = generate_dummy_data()
    
    # Generate requested plots
    plots_created = []
    
    if 'track_3d' in args.plots:
        fig = visualizer.plot_track_3d(
            data['track_points'], 
            data['track_charge'],
            data['flash_center'],
            f"3D Track Visualization (Event {args.event_id})"
        )
        output_path = os.path.join(args.output, f'track_3d_event_{args.event_id}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plots_created.append(output_path)
        plt.close(fig)
    
    if 'flash_pe' in args.plots:
        fig = visualizer.plot_flash_pe_distribution(
            data['flash_pe'],
            title=f"Flash PE Distribution (Event {args.event_id})"
        )
        output_path = os.path.join(args.output, f'flash_pe_event_{args.event_id}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plots_created.append(output_path)
        plt.close(fig)
    
    if 'timing' in args.plots and 'match_scores' in data:
        # Generate dummy timing data
        flash_times = np.full(len(data['match_scores']), data['flash_time'])
        track_times = flash_times + data['time_differences']
        
        fig = visualizer.plot_timing_correlation(
            flash_times, track_times,
            f"Timing Correlation (Event {args.event_id})"
        )
        output_path = os.path.join(args.output, f'timing_event_{args.event_id}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plots_created.append(output_path)
        plt.close(fig)
    
    if 'quality' in args.plots:
        fig = visualizer.plot_match_quality_metrics(
            data['match_scores'],
            data['spatial_distances'],
            data['time_differences'],
            f"Match Quality Metrics (Event {args.event_id})"
        )
        output_path = os.path.join(args.output, f'quality_event_{args.event_id}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plots_created.append(output_path)
        plt.close(fig)
    
    if 'crt' in args.plots:
        fig = visualizer.plot_crt_correlation(
            [data['track_points']],
            data['crt_hits'],
            f"CRT Correlation (Event {args.event_id})"
        )
        output_path = os.path.join(args.output, f'crt_event_{args.event_id}.png')
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plots_created.append(output_path)
        plt.close(fig)
    
    # Print summary
    print(f"\nVisualization complete!")
    print(f"Created {len(plots_created)} plots in {args.output}/")
    for plot in plots_created:
        print(f"  - {plot}")

if __name__ == "__main__":
    main()