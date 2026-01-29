"""Canadian Airspace Models - Track Generation Library.

This package provides tools for generating simulated aircraft tracks using
Bayesian network models trained on Canadian airspace data.
"""

# Core types and data classes
from .constants import DEFAULT_OUTPUT_DIRECTORY
from .data_classes import BayesianNetworkModelData
from .types import TrackResultData

# Main generator and session
from .generator import (
    AircraftTrackGenerator,
    TrackGenerationSession,
    generate_aircraft_tracks,
    get_available_model_files,
    load_bayesian_network_model_from_file,
)

# Export functionality
from .exporters import CsvTrackResultExporter, MatlabTrackResultExporter

# Visualization
from .visualization import TrackVisualizationRenderer

# Utilities
from .utilities import generate_unique_filepath

# Legacy API support
from .legacy_api import (
    gen_track,
    generate_plot,
    get_mat_files,
    save_as_matlab,
    save_to_csv,
)

__all__ = [
    # Constants
    "DEFAULT_OUTPUT_DIRECTORY",
    # Core types
    "BayesianNetworkModelData",
    "TrackResultData",
    # Main API
    "AircraftTrackGenerator",
    "TrackGenerationSession",
    "generate_aircraft_tracks",
    "get_available_model_files",
    "load_bayesian_network_model_from_file",
    # Exporters
    "CsvTrackResultExporter",
    "MatlabTrackResultExporter",
    # Visualization
    "TrackVisualizationRenderer",
    # Utilities
    "generate_unique_filepath",
    # Legacy API
    "gen_track",
    "generate_plot",
    "get_mat_files",
    "save_as_matlab",
    "save_to_csv",
]
