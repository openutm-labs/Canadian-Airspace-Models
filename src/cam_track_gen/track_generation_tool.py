"""Aircraft Track Generation Tool using Bayesian Networks.

This module provides a backward-compatible facade that re-exports all symbols
from the refactored modular package structure.

The module follows SOLID and DRY principles with comprehensive type hints.
This file maintains backward compatibility for existing imports from
`cam_track_gen.track_generation_tool`.
"""

from __future__ import annotations

# Re-export constants
from cam_track_gen.constants import (
    DEFAULT_OUTPUT_DIRECTORY,
    AircraftPerformanceLimits,
    CutPointLabels,
    FileExportLimits,
    PhysicsConstants,
    StatisticalThresholds,
    UnitConversionConstants,
)

# Re-export types
from cam_track_gen.types import (
    AircraftCategory,
    ProbabilityDistributionSampler,
    TrackResultData,
    TrackResultExporterInterface,
    TrackValidatorInterface,
    VariableLabel,
)

# Re-export data classes
from cam_track_gen.data_classes import (
    AircraftKinematicState,
    AltitudeBoundary,
    AngularRateLimits,
    BayesianNetworkModelData,
    DynamicPerformanceLimits,
    SimulationControlParameters,
    TrackGenerationConfiguration,
    TrigonometricStateValues,
    VelocityLimits,
    VerticalRateLimits,
)

# Re-export utilities
from cam_track_gen.utilities import (
    InverseTransformDistributionSampler,
    calculate_conditional_probability_table_index,
    convert_discrete_bin_to_continuous_value,
    generate_unique_filepath,
    get_unique_filename,
    saturate_value_within_limits,
)

# Re-export dynamics
from cam_track_gen.dynamics import (
    AircraftDynamicsCalculator,
    AircraftDynamicsIntegrator,
    AircraftTrackSimulator,
)

# Re-export Bayesian network
from cam_track_gen.bayesian_network import (
    BayesianNetworkStateSampler,
    PerformanceLimitsCalculator,
    SampledDataToTrackConverter,
)

# Re-export validation
from cam_track_gen.validation import (
    ConstraintBasedTrackValidator,
)

# Re-export exporters
from cam_track_gen.exporters import (
    CsvTrackResultExporter,
    MatlabTrackResultExporter,
)

# Re-export visualization
from cam_track_gen.visualization import (
    TrackVisualizationRenderer,
)

# Re-export generator
from cam_track_gen.generator import (
    AircraftTrackGenerator,
    TrackGenerationSession,
    generate_aircraft_tracks,
    get_available_model_files,
    load_bayesian_network_model_from_file,
)

# Re-export legacy API
from cam_track_gen.legacy_api import (
    gen_track,
    generate_plot,
    get_mat_files,
    save_as_matlab,
    save_to_csv,
)

__all__ = [
    # Constants
    "DEFAULT_OUTPUT_DIRECTORY",
    "UnitConversionConstants",
    "PhysicsConstants",
    "AircraftPerformanceLimits",
    "StatisticalThresholds",
    "FileExportLimits",
    "CutPointLabels",
    # Types
    "AircraftCategory",
    "VariableLabel",
    "TrackResultData",
    "ProbabilityDistributionSampler",
    "TrackResultExporterInterface",
    "TrackValidatorInterface",
    # Data classes
    "VelocityLimits",
    "VerticalRateLimits",
    "AngularRateLimits",
    "DynamicPerformanceLimits",
    "AltitudeBoundary",
    "TrigonometricStateValues",
    "AircraftKinematicState",
    "SimulationControlParameters",
    "BayesianNetworkModelData",
    "TrackGenerationConfiguration",
    # Utilities
    "generate_unique_filepath",
    "get_unique_filename",
    "saturate_value_within_limits",
    "InverseTransformDistributionSampler",
    "calculate_conditional_probability_table_index",
    "convert_discrete_bin_to_continuous_value",
    # Dynamics
    "AircraftDynamicsCalculator",
    "AircraftDynamicsIntegrator",
    "AircraftTrackSimulator",
    # Bayesian network
    "BayesianNetworkStateSampler",
    "SampledDataToTrackConverter",
    "PerformanceLimitsCalculator",
    # Validation
    "ConstraintBasedTrackValidator",
    # Exporters
    "CsvTrackResultExporter",
    "MatlabTrackResultExporter",
    # Visualization
    "TrackVisualizationRenderer",
    # Generator
    "AircraftTrackGenerator",
    "TrackGenerationSession",
    "generate_aircraft_tracks",
    "get_available_model_files",
    "load_bayesian_network_model_from_file",
    # Legacy API
    "gen_track",
    "generate_plot",
    "get_mat_files",
    "save_as_matlab",
    "save_to_csv",
]
