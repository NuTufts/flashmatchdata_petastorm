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

# ROOT file handling
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)  # Run ROOT in batch mode
    HAS_ROOT = True
except ImportError:
    print("Warning: ROOT not available. Using dummy data.")
    HAS_ROOT = False

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
                self.flashmatch_tree = self.root_file.Get("FlashMatchData")
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
                            track_v = self.flashmatch_tree.track_v
                            if track_v.size() > 0:
                                track = track_v.at(0)
                                print(f"Track object methods: {[method for method in dir(track) if not method.startswith('_')]}")
                        except Exception as e:
                            print(f"Could not inspect track: {e}")
                        
                        try:
                            opflash_v = self.flashmatch_tree.opflash_v
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
        if not self.flashmatch_tree:
            return self._generate_dummy_tracks()
        
        try:
            if entry >= self.flashmatch_tree.GetEntries():
                return self._generate_dummy_tracks()
            
            # Get the entry
            self.flashmatch_tree.GetEntry(entry)
            
            # Access the track vector branch
            track_v = self.flashmatch_tree.track_v
            
            all_tracks = []
            for i in range(track_v.size()):
                track = track_v.at(i)
                
                # Extract trajectory points
                try:
                    n_points = track.NumberTrajectoryPoints()
                    if n_points > 0:
                        points = []
                        for j in range(n_points):
                            pos = track.LocationAtPoint(j)
                            points.append([pos.X(), pos.Y(), pos.Z()])
                        
                        # Try different method names for start/end positions
                        try:
                            start_pos = track.Start()
                            start = [start_pos.X(), start_pos.Y(), start_pos.Z()]
                        except AttributeError:
                            try:
                                start_pos = track.Vertex()
                                start = [start_pos.X(), start_pos.Y(), start_pos.Z()]
                            except AttributeError:
                                start = points[0] if points else [0, 0, 0]
                        
                        try:
                            end_pos = track.End()
                            end = [end_pos.X(), end_pos.Y(), end_pos.Z()]
                        except AttributeError:
                            end = points[-1] if points else [0, 0, 0]
                        
                        try:
                            length = track.Length()
                        except AttributeError:
                            # Calculate length from points if method doesn't exist
                            if len(points) > 1:
                                length = np.sum([np.linalg.norm(np.array(points[k+1]) - np.array(points[k])) 
                                               for k in range(len(points)-1)])
                            else:
                                length = 0.0
                        
                        track_data = {
                            'points': np.array(points),
                            'start': start,
                            'end': end,
                            'length': length,
                            'type': 'cosmic',  # All tracks in FlashMatchData are cosmic tracks
                            'track_id': i
                        }
                        all_tracks.append(track_data)
                except Exception as e:
                    print(f"Error processing track {i}: {e}")
                    continue
            
            return all_tracks if all_tracks else self._generate_dummy_tracks()
            
        except Exception as e:
            print(f"Error loading tracks from FlashMatchData tree: {e}")
            return self._generate_dummy_tracks()
    
    def load_flash_data(self, entry: int = 0, flash_types: List[str] = None) -> List[Dict]:
        """Load optical flash data from FlashMatchData tree"""
        if not self.flashmatch_tree:
            return self._generate_dummy_flashes()
        
        try:
            if entry >= self.flashmatch_tree.GetEntries():
                return self._generate_dummy_flashes()
            
            # Get the entry
            self.flashmatch_tree.GetEntry(entry)
            
            # Access the opflash vector branch
            opflash_v = self.flashmatch_tree.opflash_v
            
            all_flashes = []
            for i in range(opflash_v.size()):
                flash = opflash_v.at(i)
                
                try:
                    # Get PMT PE values
                    pe_values = []
                    for pmt_id in range(32):  # MicroBooNE has 32 PMTs
                        try:
                            pe_values.append(flash.PE(pmt_id))
                        except:
                            pe_values.append(0.0)  # Default if PE method fails
                    
                    # Try to get flash time
                    try:
                        flash_time = flash.Time()
                    except AttributeError:
                        flash_time = 0.0
                    
                    # Try to get total PE
                    try:
                        total_pe = flash.TotalPE()
                    except AttributeError:
                        total_pe = sum(pe_values)
                    
                    # Try to get flash center - use YCenter and ZCenter methods
                    try:
                        y_center = flash.YCenter()
                        z_center = flash.ZCenter()
                        center = [128.0, y_center, z_center]  # X at detector center, Y and Z from flash
                    except AttributeError:
                        center = [128.0, 0.0, 500.0]  # Default detector center
                    
                    flash_data = {
                        'time': flash_time,
                        'total_pe': total_pe,
                        'pe_per_pmt': np.array(pe_values),
                        'center': center,
                        'type': 'flash',  # All flashes in FlashMatchData
                        'flash_id': i
                    }
                    all_flashes.append(flash_data)
                    
                except Exception as e:
                    print(f"Error processing flash {i}: {e}")
                    continue
            
            return all_flashes if all_flashes else self._generate_dummy_flashes()
            
        except Exception as e:
            print(f"Error loading flashes from FlashMatchData tree: {e}")
            return self._generate_dummy_flashes()
    
    def get_entry_count(self) -> int:
        """Get number of entries in the file"""
        if not self.flashmatch_tree:
            return 10  # Dummy data has 10 entries
        
        return self.flashmatch_tree.GetEntries()
    
    def _generate_dummy_tracks(self) -> List[Dict]:
        """Generate dummy track data for testing"""
        np.random.seed(42)
        tracks = []
        
        n_tracks = np.random.randint(2, 8)
        for i in range(n_tracks):
            # Generate a track that goes through the detector
            start = np.random.uniform([10, -100, 50], [240, 100, 950])
            direction = np.random.uniform([-1, -1, -1], [1, 1, 1])
            direction = direction / np.linalg.norm(direction)
            
            length = np.random.uniform(50, 500)
            n_points = int(length / 5) + 1
            
            points = []
            for j in range(n_points):
                point = start + j * 5 * direction
                points.append(point)
            
            track_type = np.random.choice(['boundarycosmicreduced', 'containedcosmicreduced', 'cosmictrack'])
            
            tracks.append({
                'points': np.array(points),
                'start': start.tolist(),
                'end': (start + length * direction).tolist(),
                'length': length,
                'type': track_type,
                'track_id': i
            })
        
        return tracks
    
    def _generate_dummy_flashes(self) -> List[Dict]:
        """Generate dummy flash data for testing"""
        np.random.seed(42)
        flashes = []
        
        n_flashes = np.random.randint(1, 5)
        for i in range(n_flashes):
            # Generate PMT PE values
            pe_values = np.random.poisson(10, 32)
            total_pe = np.sum(pe_values)
            
            # Generate flash center
            center = [128.0, 0.0, np.random.uniform(100, 900)]
            
            # Generate flash time
            flash_time = np.random.uniform(-5, 15)  # μs
            
            flash_type = np.random.choice(['simpleFlashBeam', 'simpleFlashCosmic'])
            
            flashes.append({
                'time': flash_time,
                'total_pe': total_pe,
                'pe_per_pmt': pe_values,
                'center': center,
                'type': flash_type,
                'flash_id': i
            })
        
        return flashes

class CosmicDashboard:
    """Main dashboard class for cosmic ray visualization"""
    
    def __init__(self, data_loader: CosmicDataLoader):
        self.data_loader = data_loader
        # Use default Dash styling
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
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
            # Timing correlation plot
            dcc.Graph(id='timing-correlation-plot', style={'height': '2000px', 'width': '100%'}),
            
            # Track selection controls
            html.Div([
                html.Label("Select Track for 3D View:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='track-selection-dropdown',
                    placeholder="Select a track...",
                    clearable=False,
                    style={'marginBottom': '10px'}
                )
            ], style={'marginTop': '20px', 'marginBottom': '20px'}),
            
            # 3D track visualization
            dcc.Graph(id='track-3d-plot', style={'height': '600px', 'width': '100%'})
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
             Output('track-selection-dropdown', 'value')],
            [Input('update-button', 'n_clicks')],
            [State('entry-dropdown', 'value')]
        )
        def update_timing_plot(n_clicks, entry):
            # Only update if button was clicked and entry is selected
            if n_clicks is None or entry is None:
                # Return empty figure on initial load
                return self._create_empty_figure(), [], None
            
            # Load data from FlashMatchData tree
            tracks = self.data_loader.load_cosmic_tracks(entry)
            flashes = self.data_loader.load_flash_data(entry)
            
            # Create track selection options
            track_options = [
                {'label': f'Track {i} (Length: {track["length"]:.1f} cm)', 'value': i}
                for i, track in enumerate(tracks)
            ]
            
            # Create the timing correlation plot
            timing_fig = self.create_timing_correlation_plot(tracks, flashes, entry)
            
            # Set default track selection to first track
            default_track = 0 if tracks else None
            
            return timing_fig, track_options, default_track
        
        @self.app.callback(
            Output('track-3d-plot', 'figure'),
            [Input('track-selection-dropdown', 'value'),
             Input('timing-correlation-plot', 'clickData')],
            [State('entry-dropdown', 'value')]
        )
        def update_3d_plot(selected_track, click_data, entry):
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
            
            # Load tracks if we have an entry and track selection
            if entry is not None and track_to_show is not None:
                tracks = self.data_loader.load_cosmic_tracks(entry)
                if 0 <= track_to_show < len(tracks):
                    return self.create_3d_track_plot(tracks[track_to_show], entry, track_to_show)
            
            # Return empty 3D plot
            return self._create_empty_3d_figure()
    
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
                "aspectratio": {"x": 1, "y": 1, "z": 4},
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
        
        # Constants
        DRIFT_VELOCITY = 0.109  # cm/μs
        
        # Plot cosmic tracks
        for i, track in enumerate(tracks):
            points = track['points']  # Array of [x, y, z] points
            
            # Convert X coordinates to time using drift velocity
            z_coords = points[:, 2]  # Z coordinates for x-axis
            times = points[:, 0] / DRIFT_VELOCITY  # Convert X to time for y-axis
            
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
                
            z_width = 50.0  # Width of flash line in Z direction
            
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
    
    def create_3d_track_plot(self, track: Dict, entry: int = None, track_id: int = None) -> go.Figure:
        """Create 3D visualization of a single track with detector outline"""
        fig = go.Figure()
        
        # Add the track trajectory
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
        
        # Add detector outline
        self._add_detector_outline_3d(fig)
        
        # Update layout
        det3d_layout = self._get_default_det3d_layout()
        title = f"3D Track {track_id} - Event {entry}" if track_id is not None and entry is not None else "3D Track Visualization"
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
    
    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        """Run the dashboard"""
        print(f"Starting Cosmic Ray Timing Dashboard at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)

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
    
    # Create and run dashboard
    dashboard = CosmicDashboard(data_loader)
    dashboard.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
