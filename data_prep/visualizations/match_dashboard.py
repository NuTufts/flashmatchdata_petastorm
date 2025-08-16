#!/usr/bin/env python3
"""
Cosmic Ray Reconstruction Dashboard

Interactive Plotly Dash application for visualizing cosmic ray reconstruction data.
This provides a comprehensive view of cosmic tracks, optical flashes, and their correlations.

Usage:
    python cosmic_dashboard.py --input cosmic_reco_output.root --port 8050
    
Author: Generated with Claude Code
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# Data handling
import numpy as np
import pandas as pd

# Plotly and Dash
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import lardly
from lardly.data.larlite_opflash import visualize_larlite_opflash_3d, visualize_empty_opflash

try:
    from lardly.crtoutline import CRTOutline
    crtdetector = CRTOutline()
except:
    print("Could not import CRT outline")
    crtdetector = None

# ROOT file handling
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)  # Run ROOT in batch mode
    HAS_ROOT = True
except ImportError:
    print("Warning: ROOT not available. Using dummy data.")
    HAS_ROOT = False

class OpFlash:
    """ 
    Class with methods that mimic the larlite::opflash object. 
    We define this so we can pass
    """
    def __init__(self,pe_per_pmt):
        self.pe_per_pmt = {}
        self.nopdets = 32
        for iopdet in range(self.nopdets):
            self.pe_per_pmt[iopdet] = pe_per_pmt[iopdet]
    
    def nOpDets(self):
        return self.nopdets

    def PE(self,ipmt):
        return self.pe_per_pmt[ipmt]

DRIFT_VELOCITY = 0.109 # cm/μs
    

class CosmicDataLoader:
    """Class to handle loading and processing cosmic ray reconstruction data"""
    
    def __init__(self, input_file: str = None):
        self.input_file = input_file
        self.root_file = None
        self.flashmatch_tree = None
        self.detector_bounds = {
            'x': (0.0, 256.4),
            'y': (-116.5, 116.5), 
            'z': (0.0, 1036.8)
        }
        
        if input_file and HAS_ROOT:
            self._open_file()
    
    def _open_file(self):
        """Open ROOT file and get FlashMatchData tree"""
        try:
            if os.path.exists(self.input_file):
                self.root_file = ROOT.TFile(self.input_file, "READ")
                print(f"Opened ROOT file: {self.input_file}")
                
                # Get the FlashMatchData tree
                self.flashmatch_tree = self.root_file.Get("flashmatch")
                if self.flashmatch_tree:
                    print(f"Found FlashMatchData tree with {self.flashmatch_tree.GetEntries()} entries")
                    # Print available branches for debugging
                    print("Available branches:")
                    for branch in self.flashmatch_tree.GetListOfBranches():
                        print(f"  - {branch.GetName()}")
                    
                    # Try to inspect the first entry to understand object structure
                    if self.flashmatch_tree.GetEntries() > 0:
                        self.flashmatch_tree.GetEntry(0)
                        print("\nInspecting first entry:")
                        try:
                            track_v = self.flashmatch_tree.track_segments_v
                            if track_v.size() > 0:
                                track = track_v.at(0)
                                print(f"Track object methods: {[method for method in dir(track) if not method.startswith('_')]}")
                        except Exception as e:
                            print(f"Could not inspect track: {e}")
                        
                        try:
                            opflash_v = self.flashmatch_tree.opflash_pe_v
                            if opflash_v.size() > 0:
                                flash = opflash_v.at(0)
                                print(f"Flash object methods: {[method for method in dir(flash) if not method.startswith('_')]}")
                        except Exception as e:
                            print(f"Could not inspect flash: {e}")
                else:
                    print("Warning: FlashMatchData tree not found in file")
            else:
                print(f"Error: File {self.input_file} does not exist")
                
        except Exception as e:
            print(f"Error opening ROOT file: {e}")
    
    def get_available_trees(self) -> List[str]:
        """Get list of available trees in the file"""
        trees = []
        
        if self.root_file:
            keys = self.root_file.GetListOfKeys()
            for key in keys:
                obj = key.ReadObj()
                if obj.InheritsFrom("TTree"):
                    trees.append(key.GetName())
        
        return trees
    
    def load_cosmic_tracks(self, entry: int = 0, track_types: List[str] = None) -> List[Dict]:
        """Load cosmic track data from FlashMatchData tree"""
        print("Beginning load_cosmic_tracks function")
        
        try:
            
            # Get the entry
            self.flashmatch_tree.GetEntry(entry)
            
            # Access the track vector branch
            track_v      = self.flashmatch_tree.track_segments_v
            track_hits_v = self.flashmatch_tree.track_hitpos_v
            track_sce_v  = self.flashmatch_tree.track_sce_segpts_v
            voxel_ave_vv = self.flashmatch_tree.voxel_avepos_vv
            voxel_q_vv   = self.flashmatch_tree.voxel_planecharge_vv
            
            all_tracks = []

            # Extract trajectory points
            try:
                n_points = track_v.size()
                if n_points > 0:
                    points = []
                    sce_points = []
                    vox_ave = []
                    vox_q   = []
                    for j in range(n_points):
                        pos = track_v.at(j)
                        points.append([pos[0],pos[1],pos[2]])
                        #print([pos[0],pos[1],pos[2]])

                        sce_pos = track_sce_v.at(j)
                        sce_points.append( [sce_pos[0],sce_pos[1],sce_pos[2]] )
                    
                    for j in range(voxel_ave_vv.size()):
                        vox_ave.append( [voxel_ave_vv.at(j).at(ii) for ii in range(3)] )
                        vox_q.append( [voxel_ave_vv.at(j).at(ii) for ii in range(3)] )
                    
                    # Try different method names for start/end positions
                    start_pos = points[0]
                    start = [start_pos[0], start_pos[1], start_pos[2]]
                    
                    end_pos = points[-1]
                    end = [end_pos[0], end_pos[1], end_pos[2]]
                    
                    track_data = {
                        'points': np.array(points),
                        'sce_points':np.array(sce_points),
                        'voxel_avepos':np.array(vox_ave),
                        'voxel_q':np.array(vox_q),
                        'start': start,
                        'end': end,
                        'type': 'cosmic',  # All tracks in FlashMatchData are cosmic tracks
                        'length':0.0,
                        'track_id': entry
                    }
                    all_tracks.append(track_data)
            except Exception as e:
                print(f"Error processing track {entry}: {e}")
        
            return all_tracks
            
        except Exception as e:
            print(f"Error loading tracks from FlashMatchData tree: {e}")
            return []
    
    def load_flash_data(self, entry: int = 0, flash_types: List[str] = None) -> List[Dict]:
        """Load optical flash data from FlashMatchData tree"""
        
        try:
            
            # Get the entry
            self.flashmatch_tree.GetEntry(entry)
            
            # Access the opflash vector branch
            opflash_v    = self.flashmatch_tree.opflash_pe_v
            
            all_flashes = []
            nopdet = opflash_v.size()
            try:
                # Get PMT PE values
                # MicroBooNE has 32 PMTs
                # But we actually save the 4 copies of each PMT signal
                #   1. (channels 0-31): unbiased readout (for the beam)
                #   2. (channels 100-131): reduced unbiased readout where signal is reduced to 25% (for PMTs with large signals)
                #   3. (channels 200-231): cosmic readout (for out-of-time)
                #   4. (channels 300-331): reduced cosmic readout where signal is reduced by 25%
                # The information in (1) and (2) are combined and saved into channels 0-31
                #   while the information in (3) and (4) are combined and saved into channels 200-231
                # We look into both, favoring the intime readout if both exists
                pe_values = []
                intime_sum = 0.0
                for pmt_id in range(32):  
                    try:
                        pe_values.append(opflash_v[pmt_id])
                        intime_sum += opflash_v[pmt_id]
                    except:
                        pe_values.append(0.0)  # Default if PE method fails
                
                # Try to get flash time
                try:
                    flash_time = self.flashmatch_tree.opflash_time
                except AttributeError:
                    flash_time = 0.0
                
                # Try to get total PE
                try:
                    total_pe = intime_sum
                except AttributeError:
                    total_pe = sum(pe_values)
                
                # Try to get flash center - use YCenter and ZCenter methods
                try:
                    y_center = self.flashmatch_tree.opflash_center[1]
                    z_center = self.flashmatch_tree.opflash_center[2]
                    center = [-10.0, y_center, z_center]  # X at detector center, Y and Z from flash
                except AttributeError:
                    center = [-10.0, 0.0, 500.0]  # Default detector center
                #print(f"opflash[{i}] tot={total_pe} nopdets={nopdet} intime_sum={intime_sum} cosmic_sum={cosmic_sum}")

                # Try to get flash width
                try:
                    z_width = self.flashmatch_tree.opflash_z_width
                except AttributeError:
                    z_width = 0.0

                pe_array = np.array(pe_values)
                
                flash_data = {
                    'time': flash_time,
                    'total_pe': total_pe,
                    'pe_per_pmt': pe_array,
                    'center': center,
                    'zwidth':z_width,
                    'type': 'flash',  # All flashes in FlashMatchData
                    'flash_id': entry
                }
                all_flashes.append(flash_data)
                
            except Exception as e:
                print(f"Error processing flash {entry}: {e}")
        
            return all_flashes if all_flashes else []
            
        except Exception as e:
            print(f"Error loading flashes from FlashMatchData tree: {e}")
            return []

    def load_crt_data(self, entry: int = 0 ) -> List[Dict]:
        """ Load CRT information from event"""
        
        # Get the entry
        self.flashmatch_tree.GetEntry(entry)

        ncrtpts = self.flashmatch_tree.crtmatch_endpts_v.size()
        crtdata = []
        for ipt in range(ncrtpts):
            crthit = self.flashmatch_tree.crtmatch_endpts_v.at(ipt)
            crtpoint = {
                "pos":[crthit[0],crthit[1],crthit[2]]
            }
            crtdata.append( crtpoint )

        return crtdata

    
    def get_entry_count(self) -> int:
        """Get number of entries in the file"""
        if not self.flashmatch_tree:
            return 10  # Dummy data has 10 entries
        
        return self.flashmatch_tree.GetEntries()

class CosmicDashboard:
    """Main dashboard class for cosmic ray visualization"""
    
    def __init__(self, data_loader: CosmicDataLoader):
        self.data_loader = data_loader
        # Use default Dash styling
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

        self.MATCH_TYPE_NAMES = {-1:"undefined match",
            0:"Anode Match",
            1:"Cathode Match",
            2:"CRT Track Match",
            3:"CRT Hit Match",
            4:"Track-to-flash Match"}
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        
        # Define styles
        header_style = {
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'marginBottom': '20px',
            'borderBottom': '2px solid #dee2e6'
        }
        
        control_style = {
            'padding': '15px',
            'marginBottom': '20px',
            'backgroundColor': '#ffffff',
            'border': '1px solid #dee2e6',
            'borderRadius': '5px'
        }
        
        button_style = {
            'backgroundColor': '#007bff',
            'color': 'white',
            'padding': '10px 20px',
            'border': 'none',
            'borderRadius': '4px',
            'cursor': 'pointer',
            'fontSize': '14px'
        }
        
        # Header
        header = html.Div([
            html.H1("Cosmic Ray Reconstruction Dashboard", 
                   style={'color': '#007bff', 'marginBottom': '10px'}),
            html.P("Interactive visualization of cosmic ray tracks and optical flashes",
                   style={'fontSize': '18px', 'color': '#6c757d'})
        ], style=header_style)
        
        # Controls
        controls = html.Div([
            html.Div([
                # Left column - Event selection
                html.Div([
                    html.Label("Select Event Entry:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='entry-dropdown',
                        options=[
                            {'label': f'Event {i}', 'value': i} 
                            for i in range(self.data_loader.get_entry_count())
                        ],
                        value=0,
                        placeholder="Select an event...",
                        clearable=False,
                        style={'marginBottom': '10px'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '3%'}),
                
                # Middle column - Update button
                html.Div([
                    html.Label("Update Visualization:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    html.Br(),
                    html.Button(
                        "Load Event Data",
                        id='update-button',
                        style=button_style
                    )
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '3%'}),
                
                # Right column - Info
                html.Div([
                    html.Label("Flash Match Data", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    html.P(f"Total Events: {self.data_loader.get_entry_count()}", 
                          style={'color': '#6c757d', 'margin': '0'}),
                    html.P("Select event and click button to load", 
                          style={'color': '#17a2b8', 'fontSize': '12px', 'margin': '0'})
                ], style={'width': '39%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ])
        ], style=control_style)
        
        # Main content - Timing plot and 3D track view
        main_content = html.Div([
            
            # Track selection controls
            html.Div([
                html.Label("Select Track for 3D View:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='track-selection-dropdown',
                    placeholder="Select a track...",
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ], style={'marginTop': '20px', 'marginBottom': '10px'}),
            
            # Flash selection controls
            html.Div([
                html.Label("Select Flash for 3D View:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='flash-selection-dropdown',
                    placeholder="Select a flash...",
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ], style={'marginBottom': '20px'}),
            
            # 3D track visualization
            dcc.Graph(id='track-3d-plot', style={'height': '600px', 'width': '100%'}),

            # Timing correlation plot
            dcc.Graph(id='timing-correlation-plot', style={'height': '2000px', 'width': '100%'})
        ], style={'marginBottom': '20px'})
        
        # Footer
        footer = html.Div([
            html.Hr(),
            html.P("Generated with Claude Code | MicroBooNE Cosmic Ray Analysis", 
                  style={'textAlign': 'center', 'color': '#6c757d', 'marginTop': '20px'})
        ])
        
        self.app.layout = html.Div([
            header, 
            controls, 
            main_content, 
            footer
        ], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1200px', 'padding': '0 20px'})
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('timing-correlation-plot', 'figure'),
             Output('track-selection-dropdown', 'options'),
             Output('track-selection-dropdown', 'value'),
             Output('flash-selection-dropdown', 'options'),
             Output('flash-selection-dropdown', 'value')],
            [Input('update-button', 'n_clicks')],
            [State('entry-dropdown', 'value')]
        )
        def update_timing_plot(n_clicks, entry):
            # Only update if button was clicked and entry is selected
            if n_clicks is None or entry is None:
                # Return empty figure on initial load
                return self._create_empty_figure(), [], None, [], None
            
            # Load data from FlashMatchData tree
            tracks  = self.data_loader.load_cosmic_tracks(entry)
            flashes = self.data_loader.load_flash_data(entry)
            
            self.data_loader.flashmatch_tree.GetEntry(entry)
            match_type = self.data_loader.flashmatch_tree.match_type
            match_name = self.MATCH_TYPE_NAMES[match_type]
            
            # Create track selection options
            track_options = [
                {'label': f'Track {i} - {match_name}', 'value': i}
                for i, track in enumerate(tracks)
            ]
            
            # Create flash selection options
            flash_options = [
                {'label': f'Flash {i} (PE: {flash["total_pe"]:.0f}, Time: {flash["time"]:.2f} μs) {match_name}', 'value': i}
                for i, flash in enumerate(flashes)
            ]
            
            # Create the timing correlation plot
            timing_fig = self.create_timing_correlation_plot(tracks, flashes, entry)
            
            # Set default track selection to first track
            default_track = 0 if tracks else None
            default_flash = 0 if flashes else None
            
            return timing_fig, track_options, default_track, flash_options, default_flash
        
        @self.app.callback(
            Output('track-3d-plot', 'figure'),
            [Input('track-selection-dropdown', 'value'),
             Input('flash-selection-dropdown', 'value'),
             Input('timing-correlation-plot', 'clickData')],
            [State('entry-dropdown', 'value')]
        )
        def update_3d_plot(selected_track, selected_flash, click_data, entry):
            if entry is None:
                return self._create_empty_3d_figure()
            
            # Determine which track to show
            track_to_show = selected_track
            
            # If user clicked on timing plot, try to extract track info
            if click_data is not None:
                curve_info = click_data.get('points', [{}])[0]
                curve_name = curve_info.get('data', {}).get('name', '')
                if curve_name.startswith('Track '):
                    try:
                        track_to_show = int(curve_name.split(' ')[1])
                    except:
                        pass
            
            # Load data
            tracks = self.data_loader.load_cosmic_tracks(entry)
            flashes = self.data_loader.load_flash_data(entry)
            crthits = self.data_loader.load_crt_data(entry)
            
            # Prepare track data
            track_data = None
            if track_to_show is not None and 0 <= track_to_show < len(tracks):
                track_data = tracks[track_to_show]
            
            # Prepare flash data
            flash_data = None
            if selected_flash is not None and 0 <= selected_flash < len(flashes):
                flash_data = flashes[selected_flash]
            
            # Create 3D plot with both track and flash if available
            return self.create_3d_track_plot(track_data, entry, track_to_show, flash_data=flash_data, flash_id=selected_flash, crthits=crthits)
    
    def _create_empty_figure(self) -> go.Figure:
        """Create empty figure for initial load"""
        fig = go.Figure()
        fig.add_annotation(
            text="Select an event and click 'Load Event Data' to view timing correlation",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title="Track-Flash Timing Correlation",
            xaxis_title="Z Position [cm]",
            yaxis_title="Time from Trigger [μs]",
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )
        return fig

    def _get_default_det3d_layout(self) -> go.Layout:

        axis_template = {
            "showbackground": True,
            "backgroundcolor": "rgb(255,255,255)",
            "gridcolor": "rgb(175, 175, 175)",
            "zerolinecolor": "rgb(175, 175, 175)"
        }

        plot3d_layout = {
            "title": "3D Track Visualization/Detector View",
            "height":800,
            "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
            "font": {"size": 12, "color": "black"},
            "showlegend": False,
            "paper_bgcolor": "rgb(255,255,255)",
            "scene": {
                "xaxis": axis_template,
                "yaxis": axis_template,
                "zaxis": axis_template,
                "aspectratio": {"x": 1, "y": 1, "z": 1},
                "camera": {"eye": {"x": -4.0, "y": 0.25, "z": 0.0},
                    "center":{"x":0.0, "y":0.0, "z":0.0},
                    "up":dict(x=0, y=1, z=0)},
                "annotations": [],
            },
        }
        return plot3d_layout
    
    def _create_empty_3d_figure(self) -> go.Figure:
        """Create empty 3D figure for initial load"""
        fig = go.Figure()
        fig.add_annotation(
            text="Select a track to view 3D trajectory",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=16)
        )

        det3d_layout = self._get_default_det3d_layout()

        fig.update_layout(det3d_layout)
        return fig
    
    def create_timing_correlation_plot(self, tracks: List[Dict], flashes: List[Dict], entry: int = None) -> go.Figure:
        """Create timing correlation plot showing tracks and flashes vs Z-position and time"""
        fig = go.Figure() 
        
        # Plot cosmic tracks
        for i, track in enumerate(tracks):

            flash = flashes[i]
            points = track['points']  # Array of [x, y, z] points
            
            # Convert X coordinates to time using drift velocity
            z_coords = points[:, 2]  # Z coordinates for x-axis
            times = points[:, 0] / DRIFT_VELOCITY  # Convert X to time for y-axis

            # we store matches with the t0 time removed.
            # so we add it back to make it fit
            times += flash['time']
            
            # Plot track points
            fig.add_trace(go.Scatter(
                x=z_coords,
                y=times,
                mode='markers+lines',
                name=f'Track {i}',
                marker=dict(size=4, color='blue', opacity=0.7),
                line=dict(color='blue', width=2, dash='solid'),
                hovertemplate=f"Track {i}<br>Z: %{{x:.1f}} cm<br>Time: %{{y:.2f}} μs<extra></extra>"
            ))
        
        # Plot flashes as horizontal lines
        for i, flash in enumerate(flashes):
            flash_time = flash['time']
            flash_center = flash['center']
            
            # Get Z position and create a horizontal line
            # flash_center is [x, y, z] where z is the Z coordinate we want
            if len(flash_center) >= 3:
                z_center = flash_center[2]
            else:
                z_center = 500.0  # Default center
                
            z_width = flash['zwidth']  # Width of flash line in Z direction
            
            z_line = [z_center - z_width, z_center + z_width]
            time_line = [flash_time, flash_time]  # Horizontal line at flash time
            
            # Flash line
            fig.add_trace(go.Scatter(
                x=z_line,
                y=time_line,
                mode='lines',
                name=f'Flash {i} (PE={flash["total_pe"]:.0f})',
                line=dict(color='red', width=3),
                hovertemplate=f"Flash {i}<br>Time: %{{y:.2f}} μs<br>PE: {flash['total_pe']:.0f}<extra></extra>"
            ))
            
            # Add flash center marker
            fig.add_trace(go.Scatter(
                x=[z_center],
                y=[flash_time],
                mode='markers',
                marker=dict(size=5, color='red', symbol='diamond'),
                name=f'Flash {i} center',
                showlegend=False,
                hovertemplate=f"Flash {i} center<br>Z: %{{x:.1f}} cm<br>Time: %{{y:.2f}} μs<extra></extra>"
            ))

            # Flash line: timing if cathod crossing
            flash_time_cathode = flash_time + 256.0/0.109 # add 2348.6 usec to time
            time_line_cathode = [flash_time_cathode,flash_time_cathode]
            fig.add_trace(go.Scatter(
                x=z_line,
                y=time_line_cathode,
                mode='lines',
                name=f'Flash {i} (PE={flash["total_pe"]:.0f})',
                line=dict(color='red', width=3, dash='dash'),
                hovertemplate=f"Flash {i} (cathode)<br>Time: %{{y:.2f}} μs<br>PE: {flash['total_pe']:.0f}<extra></extra>"
            ))
            
            # Add flash center marker
            fig.add_trace(go.Scatter(
                x=[z_center],
                y=[flash_time_cathode],
                mode='markers',
                marker=dict(size=5, color='red', symbol='diamond'),
                name=f'Flash {i} center',
                showlegend=False,
                hovertemplate=f"Flash {i} (cathode) center<br>Z: %{{x:.1f}} cm<br>Time: %{{y:.2f}} μs<extra></extra>"
            ))
        
        # Add TPC readout window bounds as horizontal dashed lines
        z_range = [0, 1037]  # Full detector Z range
        
        # Lower bound: -400 μs
        fig.add_trace(go.Scatter(
            x=z_range,
            y=[-400, -400],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='TPC Readout Window',
            hovertemplate="TPC Start: -400 μs<extra></extra>",
            showlegend=True
        ))
        
        # Upper bound: +2635 μs
        fig.add_trace(go.Scatter(
            x=z_range,
            y=[2635, 2635],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='TPC Window End',
            hovertemplate="TPC End: +2635 μs<extra></extra>",
            showlegend=False  # Don't duplicate in legend
        ))
        
        # Update layout
        title = f"Track-Flash Timing Correlation - Event {entry}" if entry is not None else "Track-Flash Timing Correlation"
        
        fig.update_layout(
            title=title,
            xaxis_title="Z Position [cm]",
            yaxis_title="Time from Trigger [μs]",
            xaxis=dict(
                range=[0, 1037],  # MicroBooNE Z range
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray'
            ),
            hovermode='closest',
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            width=1100,
            height=2000
        )
        
        # Add annotation explaining the conversion
        fig.add_annotation(
            text=f"Track timing calculated from X-coordinate using drift velocity = {DRIFT_VELOCITY} cm/μs",
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            showarrow=False,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10)
        )
        
        return fig
    
    def create_3d_track_plot(self, track: Dict = None, entry: int = None, track_id: int = None, 
                           flash_data: Dict = None, flash_id: int = None, crthits: List = None) -> go.Figure:
        """Create 3D visualization of a single track or flash with detector outline"""
        fig = go.Figure()
        
        if flash_data is not None:
            # Visualize optical flash with PMT signals
            self._add_flash_visualization(fig, flash_data, flash_id)
            title = f"3D Flash {flash_id} - Event {entry}" if flash_id is not None and entry is not None else "3D Flash Visualization"
        
        if track is not None:
            # Add the track trajectory
            # print("Add the track trajectory")
            points = track['points']
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1], 
                z=points[:, 2],
                mode='lines+markers',
                name=f'Track {track_id}' if track_id is not None else 'Track',
                line=dict(color='blue', width=4),
                marker=dict(size=3, color='blue'),
                hovertemplate="X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Z: %{z:.1f} cm<extra></extra>"
            ))
            
            # Add track start and end points
            start, end = track['start'], track['end']
            fig.add_trace(go.Scatter3d(
                x=[start[0]], y=[start[1]], z=[start[2]],
                mode='markers',
                name='Track Start',
                marker=dict(size=8, color='green', symbol='circle'),
                hovertemplate="Start<br>X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Z: %{z:.1f} cm<extra></extra>"
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[end[0]], y=[end[1]], z=[end[2]],
                mode='markers',
                name='Track End',
                marker=dict(size=8, color='red', symbol='square'),
                hovertemplate="End<br>X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Z: %{z:.1f} cm<extra></extra>"
            ))
            title = f"3D Track {track_id} - Event {entry}" if track_id is not None and entry is not None else "3D Track Visualization"

            # Add the track trajectory - where the points have had the space charge effect removed
            # print("Add the track trajectory")
            sce_points = track['sce_points']
            fig.add_trace(go.Scatter3d(
                x=sce_points[:, 0],
                y=sce_points[:, 1], 
                z=sce_points[:, 2],
                mode='lines+markers',
                name=f'Track {track_id} (SCE corrected)' if track_id is not None else 'Track (SCE corrected)',
                line=dict(color='red', width=4),
                marker=dict(size=3, color='red'),
                hovertemplate="X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Z: %{z:.1f} cm<extra></extra>"
            ))

            voxel_avepos = track['voxel_avepos']
            voxel_q      = (track['voxel_q']-50.0)/50.0
            fig.add_trace(go.Scatter3d(
                x=voxel_avepos[:, 0],
                y=voxel_avepos[:, 1], 
                z=voxel_avepos[:, 2],
                mode='markers',
                name=f'Track {track_id} (Voxel Pts)' if track_id is not None else 'Track (SCE corrected)',
                #line=dict(color='black', width=4),
                marker=dict(size=3, color=voxel_q[:,2], cmin=0.0, cmax=10.0, colorscale='bluered'),
                hovertemplate="X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Z: %{z:.1f} cm<extra></extra>"
            ))
            print("voxel_avepos: ",voxel_avepos.shape)
            
        else:
            title = "3D Visualization"

        if crthits is not None:
            for crthit in crthits:
                pos = crthit['pos']
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode='markers',
                    name='Track End',
                    marker=dict(size=8, color='red', symbol='square'),
                    hovertemplate="CRTHIT<br>X: %{x:.1f} cm<br>Y: %{y:.1f} cm<br>Z: %{z:.1f} cm<extra></extra>"
                ))
        
        # Add detector outline
        self._add_detector_outline_3d(fig)

        # Add CRT outline
        self._add_crt_outline(fig)
        
        # Update layout
        det3d_layout = self._get_default_det3d_layout()
        det3d_layout['title'] = title
        fig.update_layout(det3d_layout)
        
        return fig
    
    def _add_detector_outline_3d(self, fig: go.Figure):
        """Add detector outline to 3D plot"""
        # MicroBooNE TPC dimensions
        x_min, x_max = 0.0, 256.4
        y_min, y_max = -116.5, 116.5
        z_min, z_max = 0.0, 1036.8
        
        # Define detector face corners
        faces = [
            # Bottom face (y = y_min)
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max], [x_min, y_min, z_min]],
            # Top face (y = y_max)
            [[x_min, y_max, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_min, y_max, z_max], [x_min, y_max, z_min]],
            # Upstream face (z = z_min)
            [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], [x_min, y_min, z_min]],
            # Downstream face (z = z_max)
            [[x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max], [x_min, y_min, z_max]],
            # Cathode face (x = x_min)
            [[x_min, y_min, z_min], [x_min, y_max, z_min], [x_min, y_max, z_max], [x_min, y_min, z_max], [x_min, y_min, z_min]],
            # Anode face (x = x_max)
            [[x_max, y_min, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_max, y_min, z_max], [x_max, y_min, z_min]]
        ]
        
        # Create coordinate arrays for all faces
        x_coords = []
        y_coords = []
        z_coords = []
        
        for face in faces:
            for point in face:
                x_coords.append(point[0])
                y_coords.append(point[1])
                z_coords.append(point[2])
            # Add None to create breaks between faces
            x_coords.append(None)
            y_coords.append(None)
            z_coords.append(None)
        
        # Add detector outline trace
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            name='Detector Outline',
            line=dict(color='black', width=2),
            hoverinfo='skip',
            showlegend=True
        ))

    def _add_crt_outline(self, fig: go.Figure ):
        if crtdetector is not None:
            crt_traces = crtdetector.getlines()
        fig.add_traces( crt_traces )
    
    def _add_flash_visualization(self, fig: go.Figure, flash_data: Dict, flash_id: int = None):
        """Add PMT visualization for optical flash"""
        # PMT positions for MicroBooNE (simplified - actual positions would be loaded from detector geometry)
        # This is a simplified representation showing PMTs on the -X (cathode) side
        
        # Get PE values for each PMT
        pe_values = flash_data['pe_per_pmt']
        max_pe = np.max(pe_values) if np.max(pe_values) > 0 else 1.0

        # Create OpFlash object for plotting
        opflashobj = OpFlash(pe_values)
        
        # Create hover text
        # hover_text = [f"PMT {i}<br>PE: {pe:.1f}<br>X: {x:.1f}<br>Y: {y:.1f}<br>Z: {z:.1f}" 
        #              for i, (pe, x, y, z) in enumerate(zip(pe_values, pmt_x, pmt_y, pmt_z))]
        

        traces = visualize_larlite_opflash_3d(opflashobj,use_opdet_index=True,use_v4_geom=True)
        fig.add_traces(traces)

    
    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        """Run the dashboard"""
        print(f"Starting Cosmic Ray Timing Dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

def export_html(data_loader: CosmicDataLoader, event_num: int, output_file: str):
    """Export timing correlation and 3D plots as standalone HTML file with interactive dropdowns"""
    import json
    
    # Load data for the specified event
    tracks = data_loader.load_cosmic_tracks(event_num)
    flashes = data_loader.load_flash_data(event_num)
    
    if not tracks and not flashes:
        print(f"Warning: No tracks or flashes found in event {event_num}")
    
    # Create a temporary dashboard instance to use its plotting methods
    temp_dashboard = CosmicDashboard(data_loader)
    
    # Create timing correlation plot
    timing_fig = temp_dashboard.create_timing_correlation_plot(tracks, flashes, event_num)
    
    # Create all 3D plot combinations
    plot_data = {}
    
    # Create plots for all tracks and flashes
    for i, track in enumerate(tracks):
        for j, flash in enumerate(flashes):
            # Create plot with both track and flash
            fig = temp_dashboard.create_3d_track_plot(track, event_num, i, flash_data=flash, flash_id=j)
            fig.update_layout(title=f"3D View - Track {i} & Flash {j} - Event {event_num}")
            plot_data[f"track_{i}_flash_{j}"] = fig.to_json()
        
        # Create plot with track only
        fig = temp_dashboard.create_3d_track_plot(track, event_num, i)
        fig.update_layout(title=f"3D View - Track {i} - Event {event_num}")
        plot_data[f"track_{i}_flash_none"] = fig.to_json()
    
    # Create plots for flashes only
    for j, flash in enumerate(flashes):
        fig = temp_dashboard.create_3d_track_plot(None, event_num, None, flash_data=flash, flash_id=j)
        fig.update_layout(title=f"3D View - Flash {j} - Event {event_num}")
        plot_data[f"track_none_flash_{j}"] = fig.to_json()
    
    # Create empty plot
    empty_fig = temp_dashboard._create_empty_3d_figure()
    plot_data["track_none_flash_none"] = empty_fig.to_json()
    
    # Create HTML content with dropdowns
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cosmic Ray Event {event_num} - Timing Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border: 1px solid #dee2e6; border-radius: 5px; }}
        .plot-container {{ margin-bottom: 30px; }}
        .info-box {{ background-color: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .controls {{ background-color: #ffffff; padding: 15px; margin: 20px 0; border: 1px solid #dee2e6; border-radius: 5px; }}
        .dropdown-container {{ display: inline-block; margin-right: 20px; }}
        label {{ font-weight: bold; margin-right: 10px; }}
        select {{ padding: 5px 10px; font-size: 14px; border: 1px solid #ced4da; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cosmic Ray Reconstruction Analysis - Event {event_num}</h1>
        <p>Interactive visualization of cosmic ray tracks and optical flashes from MicroBooNE LArTPC data</p>
        <div class="info-box">
            <strong>Event Summary:</strong><br>
            • Tracks found: {len(tracks)}<br>
            • Flashes found: {len(flashes)}<br>
            • Data source: {data_loader.input_file if data_loader.input_file else "Dummy data"}
        </div>
    </div>
    
    <div class="plot-container">
        <h2>Timing Correlation Plot</h2>
        <p>Shows the relationship between cosmic ray track timing (derived from X-position using drift velocity 0.109 cm/μs) and optical flash timing.</p>
        <div id="timing-plot" style="width:100%; height:2000px;"></div>
    </div>
    
    <div class="plot-container">
        <h2>3D Track and Flash Visualization</h2>
        <p>Three-dimensional view showing selected cosmic ray track and/or optical flash PMT signals.</p>
        
        <div class="controls">
            <div class="dropdown-container">
                <label for="track-select">Select Track:</label>
                <select id="track-select" onchange="updatePlot()">
                    <option value="none">None</option>
"""
    
    # Add track options
    for i, track in enumerate(tracks):
        html_content += f'                    <option value="{i}">Track {i} (Length: {track["length"]:.1f} cm)</option>\n'
    
    html_content += """                </select>
            </div>
            
            <div class="dropdown-container">
                <label for="flash-select">Select Flash:</label>
                <select id="flash-select" onchange="updatePlot()">
                    <option value="none">None</option>
"""
    
    # Add flash options
    for j, flash in enumerate(flashes):
        html_content += f'                    <option value="{j}">Flash {j} (PE: {flash["total_pe"]:.0f}, Time: {flash["time"]:.2f} μs)</option>\n'
    
    html_content += """                </select>
            </div>
        </div>
        
        <div id="3d-plot" style="width:100%; height:600px;"></div>
    </div>
    
    <div class="info-box">
        <h3>Physics Concepts</h3>
        <ul>
            <li><strong>Drift Time:</strong> X-coordinate converted to time using drift velocity (0.109 cm/μs)</li>
            <li><strong>TPC Readout Window:</strong> Black dashed lines show -400 to +2635 μs data window</li>
            <li><strong>Flash-Track Matching:</strong> Correlating particle timing with light production</li>
            <li><strong>PMT Response:</strong> Color-coded photoelectron signals from 32 photomultiplier tubes</li>
        </ul>
    </div>
    
    <script>
        // Store all plot data
        var plotData = """ + json.dumps(plot_data) + """;
        
        // Plot timing correlation
        var timing_data = """ + timing_fig.to_json() + """;
        Plotly.newPlot('timing-plot', timing_data.data, timing_data.layout, {responsive: true});
        
        // Function to update 3D plot based on dropdown selections
        function updatePlot() {
            var trackSelect = document.getElementById('track-select').value;
            var flashSelect = document.getElementById('flash-select').value;
            
            var plotKey = 'track_' + trackSelect + '_flash_' + flashSelect;
            
            if (plotData[plotKey]) {
                var data = JSON.parse(plotData[plotKey]);
                Plotly.newPlot('3d-plot', data.data, data.layout, {responsive: true});
            }
        }
        
        // Initialize with first track and first flash if available
        document.getElementById('track-select').value = """ + ('"0"' if tracks else '"none"') + """;
        document.getElementById('flash-select').value = """ + ('"0"' if flashes else '"none"') + """;
        updatePlot();
    </script>
    
    <footer style="margin-top: 40px; text-align: center; color: #6c757d; border-top: 1px solid #dee2e6; padding-top: 20px;">
        Generated with Claude Code | MicroBooNE Cosmic Ray Analysis
    </footer>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Cosmic Ray Track-Flash Timing Dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help="Input ROOT file from cosmic reconstruction (with FlashMatchData tree)"
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default="127.0.0.1",
        help="Host to run dashboard on"
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8050,
        help="Port to run dashboard on"
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Run in debug mode"
    )
    
    parser.add_argument(
        '--save-html',
        type=str,
        help="Save interactive plots as HTML file for specified event (e.g., --save-html event_0.html)"
    )
    
    parser.add_argument(
        '--event',
        type=int,
        default=0,
        help="Event number to export when using --save-html (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = CosmicDataLoader(args.input)
    
    # Check if we have valid data
    if args.input and not HAS_ROOT:
        print("Warning: ROOT not available, using dummy data")
    elif args.input and HAS_ROOT:
        trees = data_loader.get_available_trees()
        print(f"Available trees: {trees}")
        if 'FlashMatchData' not in trees:
            print("Warning: FlashMatchData tree not found in file")
    else:
        print("No input file specified, using dummy data for demonstration")
    
    # Handle HTML export mode
    if args.save_html:
        print(f"Exporting event {args.event} to {args.save_html}")
        export_html(data_loader, args.event, args.save_html)
        print(f"HTML file saved: {args.save_html}")
        return
    
    # Create and run dashboard
    dashboard = CosmicDashboard(data_loader)
    dashboard.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
