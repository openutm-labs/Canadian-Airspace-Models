"""Aircraft Track Generation Tool using Bayesian Networks.

This module provides functionality to generate realistic aircraft tracks
by sampling from Bayesian network models trained on historical flight data.

The module follows SOLID and DRY principles with comprehensive type hints.
"""

from __future__ import annotations

import csv
import math
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, TypedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from contextlib import AbstractContextManager

# Suppress pandas chained assignment warning
pd.options.mode.chained_assignment = None


# =============================================================================
# Constants - Organized by Domain
# =============================================================================


class UnitConversionConstants:
    """Physical unit conversion factors.

    All conversions maintain precision for aviation calculations.
    """

    KNOTS_TO_FEET_PER_SECOND: Final[float] = 1.68780972222222
    FEET_PER_MINUTE_TO_FEET_PER_SECOND: Final[float] = 1.0 / 60.0
    DEGREES_TO_RADIANS: Final[float] = math.pi / 180.0

    @classmethod
    def convert_knots_to_feet_per_second(cls, speed_knots: float) -> float:
        """Convert speed from knots to feet per second."""
        return speed_knots * cls.KNOTS_TO_FEET_PER_SECOND

    @classmethod
    def convert_feet_per_minute_to_feet_per_second(cls, rate_fpm: float) -> float:
        """Convert vertical rate from feet per minute to feet per second."""
        return rate_fpm * cls.FEET_PER_MINUTE_TO_FEET_PER_SECOND

    @classmethod
    def convert_degrees_to_radians(cls, angle_degrees: float) -> float:
        """Convert angle from degrees to radians."""
        return angle_degrees * cls.DEGREES_TO_RADIANS


class PhysicsConstants:
    """Physical constants used in aircraft dynamics calculations."""

    GRAVITATIONAL_ACCELERATION_FEET_PER_SECOND_SQUARED: Final[float] = 32.2
    SIMULATION_TIME_STEP_SECONDS: Final[float] = 0.1
    INTEGRATION_GAIN_FACTOR: Final[float] = 1.0


class AircraftPerformanceLimits:
    """Aircraft performance envelope limits."""

    MAXIMUM_BANK_ANGLE_RADIANS: Final[float] = 75.0 * UnitConversionConstants.DEGREES_TO_RADIANS
    MAXIMUM_ROLL_RATE_RADIANS_PER_SECOND: Final[float] = 0.524
    ROTORCRAFT_MAXIMUM_SPEED_FEET_PER_SECOND: Final[float] = 304.0
    FIXED_WING_MINIMUM_SPEED_FEET_PER_SECOND: Final[float] = 30.0
    DEFAULT_MINIMUM_VELOCITY_FEET_PER_SECOND: Final[float] = 1.7


class StatisticalThresholds:
    """Statistical thresholds for distribution-based calculations."""

    LOW_PERCENTILE: Final[int] = 1
    HIGH_PERCENTILE: Final[int] = 99


class FileExportLimits:
    """Limits for file export operations."""

    MAXIMUM_CSV_ROWS_PER_FILE: Final[int] = 1_000_000


# Default output directory for all generated files
DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path("output")


def generate_unique_filepath(output_directory: Path, base_name: str, extension: str) -> Path:
    """Generate a unique filepath by appending a number if file exists.

    Args:
        output_directory: Directory to save the file in.
        base_name: Base filename without extension.
        extension: File extension including the dot (e.g., '.csv').

    Returns:
        Unique filepath that does not exist.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    counter = 0
    filepath = output_directory / f"{base_name}{extension}"

    while filepath.exists():
        counter += 1
        filepath = output_directory / f"{base_name}_{counter}{extension}"

    return filepath


class CutPointLabels:
    """String constants for cut point labels in MATLAB files."""

    VERTICAL_RATE: Final[str] = "Vertical Rate"
    TURN_RATE: Final[str] = "Turn Rate"
    SPEED: Final[str] = "Speed"
    ACCELERATION: Final[str] = "Acceleration"
    ACCELERATION_LEGACY: Final[str] = "Aceleration"  # Handle typo in old files
    AIRSPACE: Final[str] = "Airspace"
    ALTITUDE: Final[str] = "Altitude"
    WEIGHT_TURBULENCE_CATEGORY: Final[str] = "WTC"


# =============================================================================
# Enumerations
# =============================================================================


class AircraftCategory(Enum):
    """Aircraft weight and type categories."""

    LIGHT = "Light"
    MEDIUM = "Medium"
    HEAVY = "Heavy"
    HELICOPTER = "Helicopter"
    GYROCOPTER = "Gyrocopter"
    ULTRALIGHT = "Ultralight"


class VariableLabel(Enum):
    """Bayesian network variable labels."""

    WEIGHT_TURBULENCE_CATEGORY = "WTC"
    AIRSPACE_CLASS = "Airspace"
    ALTITUDE = "Altitude"
    SPEED = "Speed"
    ACCELERATION = "Acceleration"
    VERTICAL_RATE = "VRate"
    TURN_RATE = "TRate"


# =============================================================================
# Type Definitions
# =============================================================================


class TrackResultData(TypedDict):
    """Type definition for track simulation results.

    All spatial units are in feet, velocities in feet/second, angles in radians.
    """

    time: NDArray[np.floating[Any]]
    north_position_feet: NDArray[np.floating[Any]]
    east_position_feet: NDArray[np.floating[Any]]
    altitude_feet: NDArray[np.floating[Any]]
    speed_feet_per_second: NDArray[np.floating[Any]]
    bank_angle_radians: NDArray[np.floating[Any]]
    pitch_angle_radians: NDArray[np.floating[Any]]
    heading_angle_radians: NDArray[np.floating[Any]]


# Legacy type alias for backward compatibility
TrackResult = TrackResultData


# =============================================================================
# Data Classes - Following Single Responsibility Principle
# =============================================================================


@dataclass(frozen=True)
class VelocityLimits:
    """Velocity constraints for aircraft simulation."""

    minimum_feet_per_second: float
    maximum_feet_per_second: float

    def clamp_velocity(self, velocity: float) -> float:
        """Clamp velocity to within limits."""
        return max(self.minimum_feet_per_second, min(velocity, self.maximum_feet_per_second - 1e-6))


@dataclass(frozen=True)
class VerticalRateLimits:
    """Vertical rate constraints for aircraft simulation."""

    minimum_feet_per_second: float
    maximum_feet_per_second: float

    def clamp_vertical_rate(self, rate: float) -> float:
        """Clamp vertical rate to within limits."""
        return max(self.minimum_feet_per_second, min(rate, self.maximum_feet_per_second))

    @property
    def maximum_magnitude_feet_per_second(self) -> float:
        """Get the maximum absolute vertical rate."""
        return max(abs(self.minimum_feet_per_second), abs(self.maximum_feet_per_second))


@dataclass(frozen=True)
class AngularRateLimits:
    """Angular rate constraints for aircraft simulation."""

    maximum_pitch_rate_radians_per_second: float
    maximum_yaw_rate_radians_per_second: float


@dataclass(frozen=True)
class DynamicPerformanceLimits:
    """Combined dynamic performance limits for aircraft simulation."""

    velocity_limits: VelocityLimits
    vertical_rate_limits: VerticalRateLimits
    angular_rate_limits: AngularRateLimits

    @classmethod
    def create_default(
        cls,
        minimum_velocity: float,
        maximum_velocity: float,
        minimum_vertical_rate: float,
        maximum_vertical_rate: float,
    ) -> DynamicPerformanceLimits:
        """Create default performance limits with standard angular rates."""
        return cls(
            velocity_limits=VelocityLimits(minimum_velocity, maximum_velocity),
            vertical_rate_limits=VerticalRateLimits(minimum_vertical_rate, maximum_vertical_rate),
            angular_rate_limits=AngularRateLimits(
                maximum_pitch_rate_radians_per_second=3.0 * UnitConversionConstants.DEGREES_TO_RADIANS,
                maximum_yaw_rate_radians_per_second=1_000_000.0,
            ),
        )


@dataclass(frozen=True)
class AltitudeBoundary:
    """Altitude boundaries for track validation."""

    minimum_altitude_feet: float
    maximum_altitude_feet: float
    validation_margin_feet: float = 500.0

    def is_within_bounds(self, altitude: float) -> bool:
        """Check if altitude is within valid bounds."""
        return self.minimum_altitude_feet <= altitude <= (self.maximum_altitude_feet + self.validation_margin_feet)


@dataclass(frozen=True)
class TrigonometricStateValues:
    """Pre-computed trigonometric values for aircraft state angles.

    Caches trigonometric calculations to avoid redundant computation
    during dynamics integration.
    """

    sine_pitch: float
    cosine_pitch: float
    tangent_pitch: float
    sine_bank: float
    cosine_bank: float
    sine_heading: float
    cosine_heading: float

    @classmethod
    def from_euler_angles(
        cls,
        pitch_radians: float,
        bank_radians: float,
        heading_radians: float,
    ) -> TrigonometricStateValues:
        """Create TrigonometricStateValues from Euler angles."""
        return cls(
            sine_pitch=float(np.sin(pitch_radians)),
            cosine_pitch=float(np.cos(pitch_radians)),
            tangent_pitch=float(np.tan(pitch_radians)),
            sine_bank=float(np.sin(bank_radians)),
            cosine_bank=float(np.cos(bank_radians)),
            sine_heading=float(np.sin(heading_radians)),
            cosine_heading=float(np.cos(heading_radians)),
        )


@dataclass
class AircraftKinematicState:
    """Complete kinematic state of an aircraft at a point in time.

    All units are in feet, feet/second, and radians unless otherwise noted.
    """

    velocity_feet_per_second: float
    north_position_feet: float
    east_position_feet: float
    altitude_feet: float
    heading_angle_radians: float
    pitch_angle_radians: float
    bank_angle_radians: float
    acceleration_feet_per_second_squared: float

    def to_output_array(self, current_time_seconds: float) -> NDArray[np.floating[Any]]:
        """Convert state to output array format for recording.

        Args:
            current_time_seconds: Current simulation time in seconds.

        Returns:
            Array containing [time, north, east, altitude, velocity, bank, pitch, heading].
        """
        return np.array(
            [
                current_time_seconds,
                self.north_position_feet,
                self.east_position_feet,
                self.altitude_feet,
                self.velocity_feet_per_second,
                self.bank_angle_radians,
                self.pitch_angle_radians,
                self.heading_angle_radians,
            ]
        )

    def compute_trigonometric_values(self) -> TrigonometricStateValues:
        """Compute cached trigonometric values for this state."""
        return TrigonometricStateValues.from_euler_angles(
            self.pitch_angle_radians,
            self.bank_angle_radians,
            self.heading_angle_radians,
        )


@dataclass(frozen=True)
class SimulationControlParameters:
    """Parameters controlling the aircraft dynamics simulation."""

    velocity_limits: VelocityLimits
    vertical_rate_limits: VerticalRateLimits
    maximum_pitch_rate_radians_per_second: float
    maximum_yaw_rate_radians_per_second: float

    @classmethod
    def from_performance_limits(cls, limits: DynamicPerformanceLimits) -> SimulationControlParameters:
        """Create simulation parameters from dynamic performance limits."""
        return cls(
            velocity_limits=limits.velocity_limits,
            vertical_rate_limits=limits.vertical_rate_limits,
            maximum_pitch_rate_radians_per_second=limits.angular_rate_limits.maximum_pitch_rate_radians_per_second,
            maximum_yaw_rate_radians_per_second=limits.angular_rate_limits.maximum_yaw_rate_radians_per_second,
        )


# =============================================================================
# Bayesian Network Model Data Structure
# =============================================================================


@dataclass
class BayesianNetworkModelData:
    """Container for Bayesian network model data.

    Encapsulates the DAGs, probability tables, and metadata from loaded MATLAB files.
    """

    initial_state_directed_acyclic_graph: NDArray[np.floating[Any]]
    transition_directed_acyclic_graph: NDArray[np.floating[Any]]
    initial_probability_tables: NDArray[np.object_]
    transition_probability_tables: NDArray[np.object_]
    discretization_cut_points: NDArray[np.object_]
    resample_rate_matrix: NDArray[np.floating[Any]]
    variable_labels: list[str] = field(default_factory=list)

    @classmethod
    def from_matlab_dictionary(cls, matlab_data: dict[str, Any]) -> BayesianNetworkModelData:
        """Create BayesianNetworkModelData from loaded MATLAB dictionary.

        Args:
            matlab_data: Dictionary containing loaded MATLAB file data.

        Returns:
            BayesianNetworkModelData instance.
        """
        initial_dag = matlab_data["DAG_Initial"]
        number_of_initial_variables = initial_dag.shape[0]

        variable_labels = cls._determine_variable_labels(number_of_initial_variables)
        cut_points = cls._ensure_airspace_in_cut_points(matlab_data["Cut_Points"])

        return cls(
            initial_state_directed_acyclic_graph=initial_dag,
            transition_directed_acyclic_graph=matlab_data["DAG_Transition"],
            initial_probability_tables=matlab_data["N_initial"],
            transition_probability_tables=matlab_data["N_transition"],
            discretization_cut_points=cut_points,
            resample_rate_matrix=matlab_data["resample_rate"],
            variable_labels=variable_labels,
        )

    @staticmethod
    def _determine_variable_labels(number_of_variables: int) -> list[str]:
        """Determine variable labels based on number of variables."""
        if number_of_variables == 6:
            return ["Airspace", "Altitude", "Speed", "Acceleration", "VRate", "TRate"]
        return ["WTC", "Airspace", "Altitude", "Speed", "Acceleration", "VRate", "TRate"]

    @staticmethod
    def _ensure_airspace_in_cut_points(cut_points: NDArray[np.object_]) -> NDArray[np.object_]:
        """Ensure Airspace entry exists in cut points for backward compatibility.

        Args:
            cut_points: Original cut points array.

        Returns:
            Cut points array with Airspace entry included.
        """
        airspace_exists = any(item[0][0] == "Airspace" for item in cut_points)

        if airspace_exists:
            return cut_points

        airspace_entry = [
            np.array(["Airspace"], dtype="<U8"),
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
        ]
        cut_points_list = cut_points.tolist()
        cut_points_list.append(airspace_entry)

        augmented_cut_points = np.empty((len(cut_points_list), 2), dtype=object)
        for index, item in enumerate(cut_points_list):
            augmented_cut_points[index, 0] = item[0]
            augmented_cut_points[index, 1] = item[1]

        return augmented_cut_points


# =============================================================================
# Track Generation Context
# =============================================================================


@dataclass
class TrackGenerationConfiguration:
    """Configuration object containing all data needed for track generation.

    Consolidates model data with derived indices and parameters.
    """

    model_data: BayesianNetworkModelData
    cut_points_variable_order: list[int]
    acceleration_variable_index: int
    turn_rate_variable_index: int
    resample_probabilities: NDArray[np.floating[Any]]
    altitude_boundary: AltitudeBoundary
    performance_limits: DynamicPerformanceLimits
    initial_topological_sort_order: list[int]
    transition_topological_sort_order: list[int]
    is_rotorcraft_type: bool = False


# =============================================================================
# Protocols - Interface Segregation Principle
# =============================================================================


class ProbabilityDistributionSampler(Protocol):
    """Protocol for probability sampling strategies."""

    def sample_from_distribution(self, probability_weights: NDArray[np.floating[Any]]) -> int:
        """Sample from a discrete probability distribution.

        Args:
            probability_weights: Array of probability weights.

        Returns:
            Index of selected outcome (0-based).
        """
        ...


class TrackResultExporterInterface(ABC):
    """Abstract base class for track result exporters.

    Follows Open/Closed Principle - open for extension, closed for modification.
    """

    @abstractmethod
    def export_tracks(
        self,
        track_results: Sequence[TrackResultData],
        output_filename_base: str,
        output_directory: Path | None = None,
    ) -> Path | list[Path]:
        """Export track results to file(s).

        Args:
            track_results: Sequence of track result dictionaries.
            output_filename_base: Base filename for output.
            output_directory: Directory to save files in (uses DEFAULT_OUTPUT_DIRECTORY if None).

        Returns:
            Path or list of paths to created file(s).
        """
        ...


class TrackValidatorInterface(ABC):
    """Abstract base class for track validation strategies."""

    @abstractmethod
    def validate_track(self, track_result: TrackResultData) -> bool:
        """Validate a generated track against constraints.

        Args:
            track_result: Track result to validate.

        Returns:
            True if track is valid, False otherwise.
        """
        ...


# =============================================================================
# Sampling Utilities - Single Responsibility Principle
# =============================================================================


class InverseTransformDistributionSampler:
    """Implements inverse transform sampling for discrete distributions."""

    @staticmethod
    def sample_from_distribution(probability_weights: NDArray[np.floating[Any]]) -> int:
        """Sample from a discrete probability distribution using inverse transform sampling.

        Args:
            probability_weights: Array of probability weights (need not be normalized).

        Returns:
            Index of selected outcome (0-based).
        """
        random_uniform_value = np.random.rand()
        cumulative_probability_sum = np.cumsum(probability_weights)
        scaled_threshold = cumulative_probability_sum[-1] * random_uniform_value
        exceeds_threshold_mask = cumulative_probability_sum >= scaled_threshold
        return int(np.argmax(exceeds_threshold_mask))


def calculate_conditional_probability_table_index(
    parent_variable_sizes: Sequence[int],
    parent_variable_values: Sequence[int],
) -> int:
    """Calculate the linear index for conditional probability tables based on parent variable states.

    This function converts multi-dimensional parent variable indices to a linear index
    for accessing conditional probability tables stored as 2D arrays.

    Args:
        parent_variable_sizes: Sizes (number of states) for each parent variable.
        parent_variable_values: Current values of parent variables (1-based indexing).

    Returns:
        Linear index for probability table lookup (1-based).
    """
    cumulative_size_product = np.cumprod([1] + list(parent_variable_sizes[:-1]))
    adjusted_parent_values = np.array(parent_variable_values) - 1
    linear_index = int(np.dot(cumulative_size_product, adjusted_parent_values)) + 1
    return linear_index


def convert_discrete_bin_to_continuous_value(
    bin_edge_boundaries: NDArray[np.floating[Any]],
    discrete_bin_index: int,
) -> float:
    """Convert discretized bin index back to continuous value using uniform sampling within bin.

    Args:
        bin_edge_boundaries: Array of bin boundaries defining the discretization.
        discrete_bin_index: Discrete bin index (0-based).

    Returns:
        Continuous value sampled uniformly from the bin.
    """
    bin_lower_bound = bin_edge_boundaries[discrete_bin_index]
    bin_upper_bound = bin_edge_boundaries[discrete_bin_index + 1]

    # Special case: if bin straddles zero, return exactly zero
    if bin_lower_bound <= 0 and bin_upper_bound >= 0:
        return 0.0

    return bin_lower_bound + (bin_upper_bound - bin_lower_bound) * np.random.rand()


# =============================================================================
# Value Saturation Utility
# =============================================================================


def saturate_value_within_limits(
    value: float,
    minimum_limit: float,
    maximum_limit: float,
) -> float:
    """Saturate a value within specified minimum and maximum limits.

    Args:
        value: Value to saturate.
        minimum_limit: Minimum allowed value.
        maximum_limit: Maximum allowed value.

    Returns:
        Value clamped to [minimum_limit, maximum_limit].
    """
    return max(minimum_limit, min(maximum_limit, value))


# =============================================================================
# Aircraft Dynamics Integration - Single Responsibility Principle
# =============================================================================


class AircraftDynamicsCalculator:
    """Calculates aircraft dynamics components for integration."""

    def __init__(self, control_parameters: SimulationControlParameters) -> None:
        """Initialize the dynamics calculator.

        Args:
            control_parameters: Simulation control parameters.
        """
        self.control_parameters = control_parameters
        self.gravity = PhysicsConstants.GRAVITATIONAL_ACCELERATION_FEET_PER_SECOND_SQUARED
        self.time_step = PhysicsConstants.SIMULATION_TIME_STEP_SECONDS
        self.maximum_bank_angle = AircraftPerformanceLimits.MAXIMUM_BANK_ANGLE_RADIANS

    def calculate_saturated_vertical_rate_command(
        self,
        commanded_vertical_rate: float,
    ) -> float:
        """Saturate vertical rate command within allowable limits."""
        return saturate_value_within_limits(
            commanded_vertical_rate,
            self.control_parameters.vertical_rate_limits.minimum_feet_per_second,
            self.control_parameters.vertical_rate_limits.maximum_feet_per_second,
        )

    def calculate_pitch_and_yaw_rates(
        self,
        state: AircraftKinematicState,
        trig_values: TrigonometricStateValues,
        acceleration_command: float,
        saturated_vertical_rate_command: float,
    ) -> tuple[float, float]:
        """Calculate base pitch and yaw angular rates.

        Returns:
            Tuple of (pitch_rate, yaw_rate) in radians per second.
        """
        safe_velocity = max(state.velocity_feet_per_second, 1.0)
        max_pitch_rate = self.control_parameters.maximum_pitch_rate_radians_per_second

        current_vertical_speed = state.velocity_feet_per_second * trig_values.sine_pitch
        desired_vertical_acceleration = (saturated_vertical_rate_command - current_vertical_speed) / self.time_step

        # Compute pitch rate (q) using coordinated flight equations
        pitch_rate_numerator = (
            desired_vertical_acceleration / trig_values.cosine_pitch
            + self.gravity * trig_values.cosine_pitch * trig_values.sine_bank**2
            - acceleration_command * trig_values.tangent_pitch
        )
        pitch_rate = pitch_rate_numerator / (safe_velocity * trig_values.cosine_bank)
        pitch_rate = saturate_value_within_limits(pitch_rate, -max_pitch_rate, max_pitch_rate)

        # Compute yaw rate (r) from coordinated turn equation
        yaw_rate = self.gravity * trig_values.sine_bank * trig_values.cosine_pitch / safe_velocity
        yaw_rate = saturate_value_within_limits(
            yaw_rate,
            -self.control_parameters.maximum_yaw_rate_radians_per_second,
            self.control_parameters.maximum_yaw_rate_radians_per_second,
        )

        return pitch_rate, yaw_rate

    def calculate_maximum_allowable_bank_angle(
        self,
        state: AircraftKinematicState,
        trig_values: TrigonometricStateValues,
        acceleration_command: float,
        saturated_vertical_rate_command: float,
    ) -> float:
        """Calculate maximum allowable bank angle based on dynamics constraints."""
        safe_velocity = max(state.velocity_feet_per_second, 1.0)
        max_pitch_rate = self.control_parameters.maximum_pitch_rate_radians_per_second

        current_vertical_speed = state.velocity_feet_per_second * trig_values.sine_pitch
        desired_vertical_acceleration = (saturated_vertical_rate_command - current_vertical_speed) / self.time_step

        vertical_acceleration_for_bank = min(
            desired_vertical_acceleration,
            safe_velocity * max_pitch_rate * trig_values.cosine_bank,
        )

        discriminant = (
            safe_velocity**2 * max_pitch_rate**2
            - 4 * self.gravity * acceleration_command * trig_values.sine_pitch
            + 4 * self.gravity * vertical_acceleration_for_bank
            + 4 * self.gravity**2 * trig_values.cosine_pitch**2
        )

        if discriminant < 0:
            return self.maximum_bank_angle

        cosine_bank_limit = (-safe_velocity * max_pitch_rate + np.sqrt(discriminant)) / (2 * self.gravity * trig_values.cosine_pitch)

        if abs(cosine_bank_limit) < 1:
            calculated_max_bank = float(np.arccos(cosine_bank_limit)) * 0.98
        else:
            calculated_max_bank = 0.0

        return min(self.maximum_bank_angle, calculated_max_bank)

    def calculate_commanded_roll_rate(
        self,
        trig_values: TrigonometricStateValues,
        heading_change_command: float,
        pitch_rate: float,
        yaw_rate: float,
    ) -> float:
        """Calculate commanded roll rate to achieve desired heading change."""
        yaw_rate_without_roll_change = (pitch_rate * trig_values.sine_bank + yaw_rate * trig_values.cosine_bank) / trig_values.cosine_pitch

        yaw_rate_error = heading_change_command - yaw_rate_without_roll_change
        return 20.0 * yaw_rate_error

    def calculate_position_rates(
        self,
        state: AircraftKinematicState,
        trig_values: TrigonometricStateValues,
    ) -> tuple[float, float, float]:
        """Calculate position rates (North, East, vertical).

        Returns:
            Tuple of (north_rate, east_rate, vertical_rate) in feet per second.
        """
        north_rate = state.velocity_feet_per_second * trig_values.cosine_pitch * trig_values.cosine_heading
        east_rate = state.velocity_feet_per_second * trig_values.cosine_pitch * trig_values.sine_heading
        vertical_rate = state.velocity_feet_per_second * trig_values.sine_pitch
        return north_rate, east_rate, vertical_rate


class AircraftDynamicsIntegrator:
    """Integrates aircraft dynamics using backwards Euler method."""

    def __init__(self, control_parameters: SimulationControlParameters) -> None:
        """Initialize the dynamics integrator.

        Args:
            control_parameters: Simulation control parameters containing dynamic limits.
        """
        self.control_parameters = control_parameters
        self.dynamics_calculator = AircraftDynamicsCalculator(control_parameters)
        self.time_step = PhysicsConstants.SIMULATION_TIME_STEP_SECONDS
        self.integration_gain = PhysicsConstants.INTEGRATION_GAIN_FACTOR
        self.maximum_roll_rate = AircraftPerformanceLimits.MAXIMUM_ROLL_RATE_RADIANS_PER_SECOND

    def integrate_single_time_step(
        self,
        current_state: AircraftKinematicState,
        acceleration_command: float,
        vertical_rate_command: float,
        heading_change_command: float,
    ) -> AircraftKinematicState:
        """Perform one time step of aircraft dynamics integration.

        Args:
            current_state: Current aircraft kinematic state.
            acceleration_command: Commanded acceleration [ft/s²].
            vertical_rate_command: Commanded vertical rate [ft/s].
            heading_change_command: Commanded change in heading rate [rad/s].

        Returns:
            Updated aircraft state after integration.
        """
        # Saturate vertical rate command
        saturated_vertical_rate = self.dynamics_calculator.calculate_saturated_vertical_rate_command(vertical_rate_command)

        # Compute trigonometric values once
        trig_values = current_state.compute_trigonometric_values()

        # Calculate angular rates
        pitch_rate, yaw_rate = self.dynamics_calculator.calculate_pitch_and_yaw_rates(
            current_state, trig_values, acceleration_command, saturated_vertical_rate
        )

        # Calculate maximum allowable bank angle
        maximum_bank = self.dynamics_calculator.calculate_maximum_allowable_bank_angle(
            current_state, trig_values, acceleration_command, saturated_vertical_rate
        )

        # Calculate roll rate
        roll_rate = self.dynamics_calculator.calculate_commanded_roll_rate(trig_values, heading_change_command, pitch_rate, yaw_rate)

        # Apply roll rate limits and bank angle constraints
        roll_rate = self._apply_roll_rate_constraints(current_state, roll_rate, maximum_bank)

        # Calculate body angular rates
        bank_rate, pitch_rate_body, yaw_rate_body = self._calculate_body_frame_angular_rates(trig_values, roll_rate, pitch_rate, yaw_rate)

        # Calculate position rates
        north_rate, east_rate, vertical_rate = self.dynamics_calculator.calculate_position_rates(current_state, trig_values)

        # Integrate state
        return self._integrate_state_variables(
            current_state,
            acceleration_command,
            bank_rate,
            pitch_rate_body,
            yaw_rate_body,
            north_rate,
            east_rate,
            vertical_rate,
        )

    def _apply_roll_rate_constraints(
        self,
        current_state: AircraftKinematicState,
        roll_rate: float,
        maximum_bank_angle: float,
    ) -> float:
        """Apply roll rate limits and bank angle constraints."""
        # Limit maximum roll rate
        constrained_roll_rate = saturate_value_within_limits(roll_rate, -self.maximum_roll_rate, self.maximum_roll_rate)

        # Limit maximum bank angle
        projected_bank_angle = current_state.bank_angle_radians + constrained_roll_rate * self.time_step

        if projected_bank_angle > maximum_bank_angle:
            constrained_roll_rate = (maximum_bank_angle - current_state.bank_angle_radians) / self.time_step
        elif projected_bank_angle < -maximum_bank_angle:
            constrained_roll_rate = (-maximum_bank_angle - current_state.bank_angle_radians) / self.time_step

        return constrained_roll_rate

    def _calculate_body_frame_angular_rates(
        self,
        trig_values: TrigonometricStateValues,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
    ) -> tuple[float, float, float]:
        """Calculate angular rates in body frame coordinates."""
        bank_rate = (
            roll_rate
            + pitch_rate * trig_values.sine_bank * trig_values.tangent_pitch
            + yaw_rate * trig_values.cosine_bank * trig_values.tangent_pitch
        )
        pitch_rate_body = pitch_rate * trig_values.cosine_bank - yaw_rate * trig_values.sine_bank
        yaw_rate_body = pitch_rate * trig_values.sine_bank / trig_values.cosine_pitch + yaw_rate * trig_values.cosine_bank / trig_values.cosine_pitch

        return bank_rate, pitch_rate_body, yaw_rate_body

    def _integrate_state_variables(
        self,
        current_state: AircraftKinematicState,
        acceleration_command: float,
        bank_rate: float,
        pitch_rate: float,
        yaw_rate: float,
        north_rate: float,
        east_rate: float,
        vertical_rate: float,
    ) -> AircraftKinematicState:
        """Perform backwards Euler integration of state variables."""
        delta_time = self.time_step * self.integration_gain

        new_velocity = current_state.velocity_feet_per_second + acceleration_command * delta_time
        new_velocity = self.control_parameters.velocity_limits.clamp_velocity(new_velocity)

        return AircraftKinematicState(
            velocity_feet_per_second=new_velocity,
            north_position_feet=current_state.north_position_feet + north_rate * delta_time,
            east_position_feet=current_state.east_position_feet + east_rate * delta_time,
            altitude_feet=current_state.altitude_feet + vertical_rate * delta_time,
            heading_angle_radians=current_state.heading_angle_radians + yaw_rate * delta_time,
            pitch_angle_radians=current_state.pitch_angle_radians + pitch_rate * delta_time,
            bank_angle_radians=current_state.bank_angle_radians + bank_rate * delta_time,
            acceleration_feet_per_second_squared=acceleration_command,
        )


# =============================================================================
# Track Simulation
# =============================================================================


class AircraftTrackSimulator:
    """Simulates aircraft tracks by integrating dynamics over time."""

    def __init__(self, control_parameters: SimulationControlParameters) -> None:
        """Initialize the track simulator.

        Args:
            control_parameters: Simulation control parameters for dynamics integration.
        """
        self.integrator = AircraftDynamicsIntegrator(control_parameters)
        self.time_step_seconds = PhysicsConstants.SIMULATION_TIME_STEP_SECONDS

    def simulate_track(
        self,
        initial_state: AircraftKinematicState,
        control_command_sequence: NDArray[np.floating[Any]],
        simulation_duration_seconds: float,
    ) -> TrackResultData:
        """Simulate aircraft track over specified duration.

        Args:
            initial_state: Initial aircraft kinematic state.
            control_command_sequence: Control commands array [time, acceleration, vertical_rate, turn_rate].
            simulation_duration_seconds: Simulation duration in seconds.

        Returns:
            Dictionary with time-series data for aircraft state.
        """
        number_of_time_steps = int(simulation_duration_seconds / self.time_step_seconds + 1)
        state_history_buffer = np.full((number_of_time_steps, 8), np.nan)

        current_state = initial_state
        state_history_buffer[0, :] = current_state.to_output_array(0.0)

        current_command_index = 1

        for step_index in range(1, number_of_time_steps):
            current_time = step_index * self.time_step_seconds

            # Find current command index
            matching_time_indices = np.where(control_command_sequence[:, 0] == current_time)[0]
            if matching_time_indices.size > 0:
                current_command_index = matching_time_indices[0]

            # Extract control commands
            acceleration_command = control_command_sequence[current_command_index, 1]
            vertical_rate_command = control_command_sequence[current_command_index, 2]
            heading_change_command = control_command_sequence[current_command_index, 3]

            # Integrate dynamics
            current_state = self.integrator.integrate_single_time_step(
                current_state,
                acceleration_command,
                vertical_rate_command,
                heading_change_command,
            )

            state_history_buffer[step_index, :] = current_state.to_output_array(current_time)

        return self._convert_buffer_to_track_result(state_history_buffer)

    @staticmethod
    def _convert_buffer_to_track_result(
        state_history_buffer: NDArray[np.floating[Any]],
    ) -> TrackResultData:
        """Convert state history buffer to TrackResultData format."""
        return {
            "time": state_history_buffer[:, 0],
            "north_position_feet": state_history_buffer[:, 1],
            "east_position_feet": state_history_buffer[:, 2],
            "altitude_feet": state_history_buffer[:, 3],
            "speed_feet_per_second": state_history_buffer[:, 4],
            "bank_angle_radians": state_history_buffer[:, 5],
            "pitch_angle_radians": state_history_buffer[:, 6],
            "heading_angle_radians": state_history_buffer[:, 7],
        }


# =============================================================================
# Bayesian Network Sampling
# =============================================================================


class BayesianNetworkStateSampler:
    """Samples from Bayesian network models to generate initial conditions and transitions."""

    def __init__(
        self,
        model_data: BayesianNetworkModelData,
        sampler: ProbabilityDistributionSampler | None = None,
    ) -> None:
        """Initialize the sampler with a Bayesian network model.

        Args:
            model_data: Loaded Bayesian network model data.
            sampler: Probability distribution sampler (defaults to inverse transform sampling).
        """
        self.model_data = model_data
        self.sampler = sampler or InverseTransformDistributionSampler()

        self.initial_state_graph = nx.from_numpy_array(model_data.initial_state_directed_acyclic_graph, create_using=nx.DiGraph)
        self.transition_graph = nx.from_numpy_array(model_data.transition_directed_acyclic_graph, create_using=nx.DiGraph)

        self.initial_topological_order = list(nx.topological_sort(self.initial_state_graph))
        self.transition_topological_order = list(nx.topological_sort(self.transition_graph))

        self.dynamic_variable_indices = sorted(self.transition_topological_order[-3:])

    def sample_initial_state_conditions(self) -> NDArray[np.int_]:
        """Sample initial conditions from the initial distribution network.

        Returns:
            Array of sampled discrete variable values (1-based indexing).
        """
        number_of_variables = len(self.initial_topological_order)
        sampled_variable_values = np.zeros(number_of_variables, dtype=int)

        parent_sizes = np.array([array[0].shape[0] for array in self.model_data.initial_probability_tables])

        for variable_index in self.initial_topological_order:
            parent_indicator_mask = self.model_data.initial_state_directed_acyclic_graph[:, variable_index]
            conditional_probability_index = 0

            if np.any(parent_indicator_mask):
                conditional_probability_index = calculate_conditional_probability_table_index(
                    parent_sizes[parent_indicator_mask == 1].tolist(),
                    sampled_variable_values[parent_indicator_mask == 1].tolist(),
                )

            probability_table = self.model_data.initial_probability_tables[variable_index][0]
            column_index = max(0, conditional_probability_index - 1)
            sampled_variable_values[variable_index] = self.sampler.sample_from_distribution(probability_table[:, column_index]) + 1

        return sampled_variable_values

    def sample_transition_state_sequence(
        self,
        initial_state_conditions: NDArray[np.int_],
        sequence_length: int,
    ) -> list[list[float]]:
        """Sample a sequence of transitions from the transition network.

        Args:
            initial_state_conditions: Initial condition values.
            sequence_length: Number of time steps to sample.

        Returns:
            List of transition values for each time step.
        """
        dynamic_variables_are_dependent = np.any(
            self.model_data.transition_directed_acyclic_graph[np.ix_(self.dynamic_variable_indices, self.dynamic_variable_indices)]
        )

        if dynamic_variables_are_dependent:
            return self._sample_dependent_variable_transitions(initial_state_conditions, sequence_length)
        return self._sample_independent_variable_transitions(initial_state_conditions, sequence_length)

    def _sample_dependent_variable_transitions(
        self,
        initial_state_conditions: NDArray[np.int_],
        sequence_length: int,
    ) -> list[list[float]]:
        """Sample transitions when dynamic variables depend on each other."""
        number_of_initial_variables = len(initial_state_conditions)
        number_of_transition_variables = len(self.transition_topological_order)

        parent_sizes = self._build_combined_parent_sizes(number_of_initial_variables)

        evidence_state_array = np.zeros(number_of_transition_variables, dtype=int)
        evidence_state_array[:number_of_initial_variables] = initial_state_conditions

        all_transition_samples: list[list[float]] = []

        for _ in range(sequence_length):
            for variable_index in self.transition_topological_order[-3:]:
                parent_indicator_mask = self.model_data.transition_directed_acyclic_graph[:, variable_index]

                if np.any(parent_indicator_mask):
                    conditional_index = calculate_conditional_probability_table_index(
                        parent_sizes[parent_indicator_mask == 1].tolist(),
                        evidence_state_array[parent_indicator_mask == 1].tolist(),
                    )
                else:
                    conditional_index = 1

                probability_table = self.model_data.transition_probability_tables[variable_index][0]
                evidence_state_array[variable_index] = self.sampler.sample_from_distribution(probability_table[:, conditional_index - 1]) + 1
                evidence_state_array[variable_index - 3] = evidence_state_array[variable_index]

            all_transition_samples.append(evidence_state_array[-3:].copy().tolist())
            evidence_state_array[-3:] = 0

        return all_transition_samples

    def _sample_independent_variable_transitions(
        self,
        initial_state_conditions: NDArray[np.int_],
        sequence_length: int,
    ) -> list[list[float]]:
        """Sample transitions when dynamic variables are independent of each other."""
        number_of_initial_variables = len(initial_state_conditions)
        number_of_transition_variables = len(self.transition_topological_order)

        parent_sizes = self._build_combined_parent_sizes(number_of_initial_variables)

        evidence_state_array = np.zeros(number_of_transition_variables, dtype=int)
        evidence_state_array[:number_of_initial_variables] = initial_state_conditions

        # Pre-compute cumulative sums and random thresholds for efficiency
        cumulative_probability_sums: dict[int, NDArray[np.floating[Any]]] = {}
        random_threshold_values = np.zeros((sequence_length, number_of_transition_variables))

        for variable_index in self.transition_topological_order[-3:]:
            parent_indicator_mask = self.model_data.transition_directed_acyclic_graph[:, variable_index]

            if np.any(parent_indicator_mask):
                conditional_index = calculate_conditional_probability_table_index(
                    parent_sizes[parent_indicator_mask == 1].tolist(),
                    evidence_state_array[parent_indicator_mask == 1].tolist(),
                )
            else:
                conditional_index = 1

            probability_table = self.model_data.transition_probability_tables[variable_index][0]
            cumulative_probability_sums[variable_index] = np.cumsum(probability_table[:, int(conditional_index) - 1])
            random_threshold_values[:, variable_index] = cumulative_probability_sums[variable_index][-1] * np.random.rand(sequence_length)

        all_transition_samples: list[list[float]] = []

        for time_step in range(sequence_length):
            for variable_index in self.transition_topological_order[-3:]:
                evidence_state_array[variable_index] = (
                    np.searchsorted(
                        cumulative_probability_sums[variable_index],
                        random_threshold_values[time_step, variable_index],
                    )
                    + 1
                )
                evidence_state_array[variable_index - 3] = evidence_state_array[variable_index]

            all_transition_samples.append(evidence_state_array[-3:].copy().tolist())
            evidence_state_array[-3:] = 0

        return all_transition_samples

    def _build_combined_parent_sizes(self, number_of_initial_variables: int) -> NDArray[np.int_]:
        """Build combined parent sizes array from initial and transition probability tables."""
        parent_sizes = np.array([array[0].shape[0] for array in self.model_data.transition_probability_tables])
        parent_sizes[:number_of_initial_variables] = [array[0].shape[0] for array in self.model_data.initial_probability_tables]
        return parent_sizes


# =============================================================================
# Track Data Processor
# =============================================================================


class SampledDataToTrackConverter:
    """Converts sampled Bayesian network data into aircraft tracks."""

    def __init__(self, configuration: TrackGenerationConfiguration) -> None:
        """Initialize the converter with generation configuration.

        Args:
            configuration: Track generation configuration containing model and parameters.
        """
        self.configuration = configuration
        self.model_data = configuration.model_data

    def convert_sampled_data_to_track(
        self,
        initial_state_conditions: dict[str, int | float],
        transition_sequence_data: list[list[float]],
        simulation_duration_seconds: int,
    ) -> TrackResultData:
        """Convert sampled data into control commands and generate aircraft track.

        Args:
            initial_state_conditions: Initial conditions dictionary.
            transition_sequence_data: Transition data sequence.
            simulation_duration_seconds: Simulation duration in seconds.

        Returns:
            Simulation results dictionary.
        """
        number_of_initial_variables = len(initial_state_conditions)
        variable_labels = self._get_variable_labels_for_model(number_of_initial_variables)
        increment_offset = 2 if number_of_initial_variables == 6 else 3

        # Convert discretized initial conditions to continuous values
        initial_continuous_values = self._convert_initial_conditions_to_continuous(initial_state_conditions, variable_labels)

        # Process transition data with resampling
        control_command_sequence = self._process_transition_data_with_resampling(
            initial_continuous_values,
            transition_sequence_data,
            simulation_duration_seconds,
            increment_offset,
        )

        # Create initial aircraft state
        initial_kinematic_state = self._create_initial_kinematic_state(initial_continuous_values, variable_labels)

        # Create simulation parameters
        simulation_parameters = self._create_simulation_control_parameters()

        # Run simulation
        simulator = AircraftTrackSimulator(simulation_parameters)
        return simulator.simulate_track(
            initial_kinematic_state,
            control_command_sequence,
            simulation_duration_seconds,
        )

    @staticmethod
    def _get_variable_labels_for_model(number_of_variables: int) -> list[str]:
        """Get variable labels based on number of variables."""
        if number_of_variables == 6:
            return ["Airspace", "Altitude", "Speed", "Acceleration", "VRate", "TRate"]
        return ["WTC", "Airspace", "Altitude", "Speed", "Acceleration", "VRate", "TRate"]

    def _convert_initial_conditions_to_continuous(
        self,
        initial_state_conditions: dict[str, float],
        variable_labels: list[str],
    ) -> NDArray[np.floating[Any]]:
        """Convert discretized initial conditions to continuous values."""
        number_of_variables = len(initial_state_conditions)
        continuous_values = np.zeros(number_of_variables)

        for index, label in enumerate(variable_labels):
            bin_edge_boundaries = (
                self.model_data.discretization_cut_points[self.configuration.cut_points_variable_order[index], 1].flatten().astype(float)
            )

            # Handle legacy scaling for acceleration and turn rate
            if index in (self.configuration.acceleration_variable_index, self.configuration.turn_rate_variable_index):
                if any(edge > 100 for edge in bin_edge_boundaries):
                    bin_edge_boundaries = bin_edge_boundaries / 100
                    self.model_data.discretization_cut_points[self.configuration.cut_points_variable_order[index], 1] = bin_edge_boundaries

            discrete_bin_value = int(np.array(initial_state_conditions[label]).item())

            if np.all(np.diff(bin_edge_boundaries) == 1):
                continuous_values[index] = discrete_bin_value
            else:
                continuous_values[index] = convert_discrete_bin_to_continuous_value(bin_edge_boundaries, discrete_bin_value - 1)

        return continuous_values

    def _process_transition_data_with_resampling(
        self,
        initial_continuous_values: NDArray[np.floating[Any]],
        transition_sequence_data: list[list[float]],
        simulation_duration_seconds: int,
        increment_offset: int,
    ) -> NDArray[np.floating[Any]]:
        """Process transition data with resampling logic."""
        transition_array = np.array(transition_sequence_data)
        resampled_data = np.column_stack((np.arange(simulation_duration_seconds), transition_array))
        resampled_data = np.array(resampled_data, dtype=float)
        resampled_data[0, 1:4] = initial_continuous_values[-3:]

        previous_state = resampled_data[0, 1:4].copy()
        indices_to_remove: list[int] = []
        resample_probabilities = self.configuration.resample_probabilities

        for step_index in range(1, len(resampled_data)):
            current_state = resampled_data[step_index, 1:4]
            state_has_changed = not np.array_equal(previous_state, current_state)

            variables_to_resample = np.where(np.random.rand(*resample_probabilities.shape) < resample_probabilities)[0] + 1

            if variables_to_resample.size > 0 or state_has_changed:
                if state_has_changed:
                    changed_variable_indices = np.where(previous_state != current_state)[0] + 1
                    variables_to_resample = np.unique(np.concatenate((variables_to_resample, changed_variable_indices)))

                previous_state = current_state.copy()
                resampled_data[step_index, 1:4] = resampled_data[step_index - 1, 1:4].copy()

                for variable_index in variables_to_resample:
                    bin_edges = self.model_data.discretization_cut_points[
                        self.configuration.cut_points_variable_order[int(variable_index) + increment_offset], 1
                    ].flatten()
                    resampled_data[step_index, variable_index] = convert_discrete_bin_to_continuous_value(
                        bin_edges, int(previous_state[int(variable_index) - 1]) - 1
                    )
            else:
                resampled_data[step_index, 1:4] = resampled_data[step_index - 1, 1:4].copy()
                indices_to_remove.append(step_index)

        control_commands = np.delete(resampled_data, indices_to_remove, axis=0)

        # Apply unit conversions to control commands
        control_commands[:, 2] *= UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND
        control_commands[:, 3] *= UnitConversionConstants.DEGREES_TO_RADIANS
        control_commands[:, 1] *= UnitConversionConstants.KNOTS_TO_FEET_PER_SECOND

        return control_commands

    def _create_initial_kinematic_state(
        self,
        continuous_values: NDArray[np.floating[Any]],
        variable_labels: list[str],
    ) -> AircraftKinematicState:
        """Create initial aircraft kinematic state from continuous values."""
        altitude_feet = continuous_values[variable_labels.index("Altitude")]
        velocity_feet_per_second = UnitConversionConstants.convert_knots_to_feet_per_second(continuous_values[variable_labels.index("Speed")])
        vertical_rate_feet_per_second = UnitConversionConstants.convert_feet_per_minute_to_feet_per_second(
            continuous_values[variable_labels.index("VRate")]
        )
        heading_change_radians_per_second = UnitConversionConstants.convert_degrees_to_radians(continuous_values[variable_labels.index("TRate")])

        # Calculate initial attitude angles
        initial_heading_radians = 0.0

        if abs(velocity_feet_per_second) < 1e-9:
            initial_pitch_radians = 0.0
        else:
            initial_pitch_radians = math.asin(vertical_rate_feet_per_second / velocity_feet_per_second)

        initial_bank_radians = math.atan(
            velocity_feet_per_second * heading_change_radians_per_second / PhysicsConstants.GRAVITATIONAL_ACCELERATION_FEET_PER_SECOND_SQUARED
        )

        return AircraftKinematicState(
            velocity_feet_per_second=velocity_feet_per_second,
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=altitude_feet,
            heading_angle_radians=initial_heading_radians,
            pitch_angle_radians=initial_pitch_radians,
            bank_angle_radians=initial_bank_radians,
            acceleration_feet_per_second_squared=UnitConversionConstants.convert_knots_to_feet_per_second(
                continuous_values[variable_labels.index(VariableLabel.ACCELERATION.value)]
            ),
        )

    def _create_simulation_control_parameters(self) -> SimulationControlParameters:
        """Create simulation control parameters from cut points."""
        cut_points = self.model_data.discretization_cut_points

        speed_cut_point_index = int(np.nonzero(cut_points[:, 0] == CutPointLabels.SPEED)[0][0])
        vertical_rate_cut_point_index = int(np.nonzero(cut_points[:, 0] == CutPointLabels.VERTICAL_RATE)[0][0])

        maximum_velocity = max(cut_points[speed_cut_point_index, 1].flatten()) * UnitConversionConstants.KNOTS_TO_FEET_PER_SECOND
        minimum_vertical_rate = (
            min(cut_points[vertical_rate_cut_point_index, 1].flatten()) * UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND
        )
        maximum_vertical_rate = (
            max(cut_points[vertical_rate_cut_point_index, 1].flatten()) * UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND
        )

        return SimulationControlParameters(
            velocity_limits=VelocityLimits(
                minimum_feet_per_second=AircraftPerformanceLimits.DEFAULT_MINIMUM_VELOCITY_FEET_PER_SECOND,
                maximum_feet_per_second=maximum_velocity,
            ),
            vertical_rate_limits=VerticalRateLimits(
                minimum_feet_per_second=minimum_vertical_rate,
                maximum_feet_per_second=maximum_vertical_rate,
            ),
            maximum_pitch_rate_radians_per_second=3.0 * UnitConversionConstants.DEGREES_TO_RADIANS,
            maximum_yaw_rate_radians_per_second=1_000_000.0,
        )


# =============================================================================
# Dynamic Limits Calculator
# =============================================================================


class PerformanceLimitsCalculator:
    """Calculates dynamic performance limits based on probability distributions."""

    def __init__(self, model_data: BayesianNetworkModelData) -> None:
        """Initialize the calculator with model data.

        Args:
            model_data: Bayesian network model data containing probability tables.
        """
        self.model_data = model_data

    def calculate_limits(
        self,
        discretization_boundaries: NDArray[np.object_],
        is_rotorcraft: bool,
    ) -> tuple[AltitudeBoundary, DynamicPerformanceLimits]:
        """Calculate dynamic limits based on probability distributions.

        Args:
            discretization_boundaries: Discretization boundaries array.
            is_rotorcraft: Whether the aircraft is a rotorcraft type.

        Returns:
            Tuple of (altitude_boundary, dynamic_performance_limits).
        """
        variable_labels = self.model_data.variable_labels
        probability_tables = self.model_data.initial_probability_tables

        altitude_index = variable_labels.index("Altitude")
        speed_index = variable_labels.index("Speed")
        vertical_rate_index = variable_labels.index("VRate")

        # Calculate altitude boundary
        altitude_boundary = AltitudeBoundary(
            minimum_altitude_feet=float(np.min(discretization_boundaries[altitude_index])),
            maximum_altitude_feet=float(np.max(discretization_boundaries[altitude_index])),
        )

        # Calculate velocity limits from distribution
        velocity_distribution = np.sum(probability_tables[speed_index, 0], axis=1)
        minimum_velocity, maximum_velocity = self._calculate_percentile_based_limits(
            velocity_distribution,
            discretization_boundaries[speed_index],
        )

        # Apply aircraft type-specific constraints
        minimum_velocity_fps = UnitConversionConstants.convert_knots_to_feet_per_second(minimum_velocity)
        maximum_velocity_fps = UnitConversionConstants.convert_knots_to_feet_per_second(maximum_velocity)

        if is_rotorcraft and maximum_velocity_fps > AircraftPerformanceLimits.ROTORCRAFT_MAXIMUM_SPEED_FEET_PER_SECOND:
            maximum_velocity_fps = AircraftPerformanceLimits.ROTORCRAFT_MAXIMUM_SPEED_FEET_PER_SECOND
        if not is_rotorcraft and minimum_velocity_fps < AircraftPerformanceLimits.FIXED_WING_MINIMUM_SPEED_FEET_PER_SECOND:
            minimum_velocity_fps = AircraftPerformanceLimits.FIXED_WING_MINIMUM_SPEED_FEET_PER_SECOND

        # Calculate vertical rate limits from distribution
        vertical_rate_distribution = np.sum(probability_tables[vertical_rate_index, 0], axis=1)
        minimum_vertical_rate, maximum_vertical_rate = self._calculate_percentile_based_limits(
            vertical_rate_distribution,
            discretization_boundaries[vertical_rate_index],
        )

        maximum_vertical_rate_fps = max(
            abs(minimum_vertical_rate) * UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND,
            abs(maximum_vertical_rate) * UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND,
        )

        if np.isnan(maximum_vertical_rate_fps):
            maximum_vertical_rate_fps = 0.0

        performance_limits = DynamicPerformanceLimits.create_default(
            minimum_velocity=minimum_velocity_fps,
            maximum_velocity=maximum_velocity_fps,
            minimum_vertical_rate=-maximum_vertical_rate_fps,
            maximum_vertical_rate=maximum_vertical_rate_fps,
        )

        return altitude_boundary, performance_limits

    @staticmethod
    def _calculate_percentile_based_limits(
        probability_distribution: NDArray[np.floating[Any]],
        bin_edge_boundaries: NDArray[np.floating[Any]],
    ) -> tuple[float, float]:
        """Calculate percentile-based limits from a probability distribution."""
        normalized_probabilities = 100 * (probability_distribution / np.sum(probability_distribution))
        cumulative_probability = np.cumsum(normalized_probabilities)

        low_percentile_index = int(np.argmax(cumulative_probability >= StatisticalThresholds.LOW_PERCENTILE))
        high_percentile_index = int(np.argmax(cumulative_probability >= StatisticalThresholds.HIGH_PERCENTILE))

        flat_boundaries = bin_edge_boundaries.flatten() if bin_edge_boundaries.ndim > 1 else bin_edge_boundaries

        return float(flat_boundaries[low_percentile_index + 1]), float(flat_boundaries[high_percentile_index + 1])


# =============================================================================
# Track Validation
# =============================================================================


class ConstraintBasedTrackValidator(TrackValidatorInterface):
    """Validates generated tracks against performance constraints."""

    def __init__(
        self,
        altitude_boundary: AltitudeBoundary,
        performance_limits: DynamicPerformanceLimits,
    ) -> None:
        """Initialize the validator.

        Args:
            altitude_boundary: Altitude boundaries for validation.
            performance_limits: Dynamic performance limits for validation.
        """
        self.altitude_boundary = altitude_boundary
        self.performance_limits = performance_limits

    def validate_track(self, track_result: TrackResultData) -> bool:
        """Check if a track satisfies all performance constraints.

        Args:
            track_result: Track result to validate.

        Returns:
            True if track is valid, False otherwise.
        """
        has_altitude_violation = self._check_altitude_constraint_violation(track_result)
        has_velocity_violation = self._check_velocity_constraint_violation(track_result)
        has_vertical_rate_violation = self._check_vertical_rate_constraint_violation(track_result)

        return not (has_altitude_violation or has_velocity_violation or has_vertical_rate_violation)

    def _check_altitude_constraint_violation(self, track_result: TrackResultData) -> bool:
        """Check for altitude constraint violations."""
        altitude_key = "altitude_feet" if "altitude_feet" in track_result else "up_ft"
        altitudes = track_result[altitude_key]  # type: ignore[literal-required]
        return bool(
            np.any(
                (altitudes < self.altitude_boundary.minimum_altitude_feet)
                | (altitudes > self.altitude_boundary.maximum_altitude_feet + self.altitude_boundary.validation_margin_feet)
            )
        )

    def _check_velocity_constraint_violation(self, track_result: TrackResultData) -> bool:
        """Check for velocity constraint violations."""
        speed_key = "speed_feet_per_second" if "speed_feet_per_second" in track_result else "speed_ftps"
        speeds = track_result[speed_key]  # type: ignore[literal-required]
        return bool(
            np.any(
                (speeds < self.performance_limits.velocity_limits.minimum_feet_per_second)
                | (speeds > self.performance_limits.velocity_limits.maximum_feet_per_second)
            )
        )

    def _check_vertical_rate_constraint_violation(self, track_result: TrackResultData) -> bool:
        """Check for vertical rate constraint violations."""
        altitude_key = "altitude_feet" if "altitude_feet" in track_result else "up_ft"
        altitudes = track_result[altitude_key]  # type: ignore[literal-required]
        times = track_result["time"]

        vertical_rate_gradient = np.gradient(altitudes, times)
        vertical_rate_magnitude = np.abs(vertical_rate_gradient)
        return bool(np.any(vertical_rate_magnitude > self.performance_limits.vertical_rate_limits.maximum_magnitude_feet_per_second))


# =============================================================================
# Main Track Generator
# =============================================================================


class AircraftTrackGenerator:
    """Main class for generating aircraft tracks from Bayesian network models.

    This class orchestrates the track generation process using dependency injection
    for all components, following the Dependency Inversion Principle.
    """

    def __init__(
        self,
        model_data: BayesianNetworkModelData,
        sampler: BayesianNetworkStateSampler | None = None,
        validator: TrackValidatorInterface | None = None,
    ) -> None:
        """Initialize the track generator.

        Args:
            model_data: Loaded Bayesian network model data.
            sampler: Optional custom sampler (created if not provided).
            validator: Optional custom validator (created if not provided).
        """
        self.model_data = model_data
        self.sampler = sampler or BayesianNetworkStateSampler(model_data)
        self._configuration = self._initialize_configuration()
        self._data_converter = SampledDataToTrackConverter(self._configuration)
        self.validator = validator or ConstraintBasedTrackValidator(
            self._configuration.altitude_boundary,
            self._configuration.performance_limits,
        )

    def _initialize_configuration(self) -> TrackGenerationConfiguration:
        """Initialize the track generation configuration."""
        number_of_initial_variables = len(self.model_data.variable_labels)

        cut_points_labels = self._get_cut_points_labels(number_of_initial_variables)
        cut_points_order = self._compute_cut_points_order(cut_points_labels)

        acceleration_label = (
            CutPointLabels.ACCELERATION_LEGACY if CutPointLabels.ACCELERATION_LEGACY in cut_points_labels else CutPointLabels.ACCELERATION
        )
        acceleration_index = cut_points_labels.index(acceleration_label)
        turn_rate_index = cut_points_labels.index(CutPointLabels.TURN_RATE)

        discretization_boundaries = self.model_data.discretization_cut_points[cut_points_order, 1]
        limits_calculator = PerformanceLimitsCalculator(self.model_data)
        altitude_boundary, performance_limits = limits_calculator.calculate_limits(discretization_boundaries, is_rotorcraft=False)

        resample_probabilities = self.model_data.resample_rate_matrix[-3:, 0]

        return TrackGenerationConfiguration(
            model_data=self.model_data,
            cut_points_variable_order=cut_points_order,
            acceleration_variable_index=acceleration_index,
            turn_rate_variable_index=turn_rate_index,
            resample_probabilities=resample_probabilities,
            altitude_boundary=altitude_boundary,
            performance_limits=performance_limits,
            initial_topological_sort_order=self.sampler.initial_topological_order,
            transition_topological_sort_order=self.sampler.transition_topological_order,
        )

    def _get_cut_points_labels(self, number_of_variables: int) -> list[str]:
        """Get cut points labels based on number of variables."""
        if number_of_variables == 6:
            # Handle legacy typo in some data files
            if self.model_data.discretization_cut_points[0, 0][0] == CutPointLabels.ACCELERATION_LEGACY:
                return [
                    CutPointLabels.AIRSPACE,
                    CutPointLabels.ALTITUDE,
                    CutPointLabels.SPEED,
                    CutPointLabels.ACCELERATION_LEGACY,
                    CutPointLabels.VERTICAL_RATE,
                    CutPointLabels.TURN_RATE,
                ]
            return [
                CutPointLabels.AIRSPACE,
                CutPointLabels.ALTITUDE,
                CutPointLabels.SPEED,
                CutPointLabels.ACCELERATION,
                CutPointLabels.VERTICAL_RATE,
                CutPointLabels.TURN_RATE,
            ]
        return [
            CutPointLabels.WEIGHT_TURBULENCE_CATEGORY,
            CutPointLabels.AIRSPACE,
            CutPointLabels.ALTITUDE,
            CutPointLabels.SPEED,
            CutPointLabels.ACCELERATION,
            CutPointLabels.VERTICAL_RATE,
            CutPointLabels.TURN_RATE,
        ]

    def _compute_cut_points_order(self, cut_points_labels: list[str]) -> list[int]:
        """Compute the order of cut points based on labels."""
        return [int(np.nonzero(self.model_data.discretization_cut_points[:, 0] == label)[0][0]) for label in cut_points_labels]

    def generate_multiple_tracks(
        self,
        number_of_tracks: int,
        simulation_duration_seconds: int,
        use_reproducible_seed: bool = False,
    ) -> tuple[list[TrackResultData], pd.DataFrame, list[list[list[float]]]]:
        """Generate multiple valid aircraft tracks.

        Args:
            number_of_tracks: Number of tracks to generate.
            simulation_duration_seconds: Duration of each track in seconds.
            use_reproducible_seed: Enable seeded generation for reproducibility.

        Returns:
            Tuple of (valid_track_results, initial_conditions_dataframe, transition_data_list).
        """
        valid_track_results: list[TrackResultData] = []
        all_initial_conditions: list[NDArray[np.int_]] = []
        all_transition_sequences: list[list[list[float]]] = []

        seed_counter = 1 if use_reproducible_seed else None
        total_valid_tracks = 0

        while total_valid_tracks < number_of_tracks:
            if use_reproducible_seed:
                np.random.seed(seed_counter)
                seed_counter = (seed_counter or 0) + 1
            else:
                np.random.seed(None)

            # Sample initial conditions
            initial_conditions = self.sampler.sample_initial_state_conditions()

            # Sample transition sequence
            transition_sequence = self.sampler.sample_transition_state_sequence(initial_conditions, simulation_duration_seconds)

            # Convert to dictionary format
            initial_conditions_dict = {label: int(initial_conditions[index]) for index, label in enumerate(self.model_data.variable_labels)}

            # Generate track
            track_result = self._data_converter.convert_sampled_data_to_track(
                initial_conditions_dict,
                transition_sequence,
                simulation_duration_seconds,
            )

            # Validate track
            if self.validator.validate_track(track_result):
                valid_track_results.append(track_result)
                all_initial_conditions.append(initial_conditions)
                all_transition_sequences.append(transition_sequence)
                total_valid_tracks += 1

        # Create DataFrame from initial conditions
        initial_conditions_dataframe = pd.DataFrame(
            all_initial_conditions,
            columns=self.model_data.variable_labels,
        )

        return valid_track_results, initial_conditions_dataframe, all_transition_sequences

    @property
    def configuration(self) -> TrackGenerationConfiguration:
        """Get the track generation configuration."""
        return self._configuration


# =============================================================================
# Result Exporters - Open/Closed Principle
# =============================================================================


class CsvTrackResultExporter(TrackResultExporterInterface):
    """Exports track results to CSV format with automatic file splitting."""

    def __init__(
        self,
        maximum_rows_per_file: int = FileExportLimits.MAXIMUM_CSV_ROWS_PER_FILE,
    ) -> None:
        """Initialize the CSV exporter.

        Args:
            maximum_rows_per_file: Maximum number of rows per output file.
        """
        self.maximum_rows_per_file = maximum_rows_per_file

    def export_tracks(
        self,
        track_results: Sequence[TrackResultData],
        output_filename_base: str,
        output_directory: Path | None = None,
    ) -> list[Path]:
        """Export track data to CSV format with automatic file splitting.

        Args:
            track_results: Sequence of track results to export.
            output_filename_base: Base filename for output.
            output_directory: Directory to save files in (uses DEFAULT_OUTPUT_DIRECTORY if None).

        Returns:
            List of paths to created files.
        """
        if not track_results:
            print("The results list is empty. No file will be saved.")
            return []

        target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
        created_files: list[Path] = []
        fieldnames = ["Aircraft_ID"] + list(track_results[0].keys())
        current_row_count = 0

        output_filepath = generate_unique_filepath(target_directory, f"{output_filename_base}_Result", ".csv")
        current_file = open(output_filepath, mode="w", newline="", encoding="utf-8")  # noqa: SIM115
        csv_writer = csv.writer(current_file)
        csv_writer.writerow(fieldnames)
        created_files.append(output_filepath)
        print(f"Writing to {output_filepath}...")

        try:
            for aircraft_id, track_result in enumerate(track_results, start=1):
                number_of_rows = len(track_result["time"])

                if current_row_count + number_of_rows > self.maximum_rows_per_file:
                    current_file.close()
                    output_filepath = generate_unique_filepath(target_directory, f"{output_filename_base}_Result", ".csv")
                    current_file = open(output_filepath, mode="w", newline="", encoding="utf-8")  # noqa: SIM115
                    csv_writer = csv.writer(current_file)
                    csv_writer.writerow(fieldnames)
                    created_files.append(output_filepath)
                    print(f"Writing to {output_filepath}...")
                    current_row_count = 0

                for row_index in range(number_of_rows):
                    row_data = [aircraft_id]
                    row_data.extend(track_result[key][row_index] for key in track_result.keys())  # type: ignore[literal-required]
                    csv_writer.writerow(row_data)

                current_row_count += number_of_rows
        finally:
            current_file.close()

        print("Data successfully saved.")
        return created_files


class MatlabTrackResultExporter(TrackResultExporterInterface):
    """Exports track results to MATLAB .mat format."""

    def export_tracks(
        self,
        track_results: Sequence[TrackResultData],
        output_filename_base: str,
        output_directory: Path | None = None,
    ) -> Path:
        """Export track results to MATLAB format.

        Args:
            track_results: Sequence of track results to export.
            output_filename_base: Base filename for output.
            output_directory: Directory to save files in (uses DEFAULT_OUTPUT_DIRECTORY if None).

        Returns:
            Path to created file.
        """
        target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
        output_filepath = generate_unique_filepath(target_directory, f"{output_filename_base}_Result", ".mat")
        scipy.io.savemat(str(output_filepath), {"Tracks": list(track_results)})
        return output_filepath


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_model_files() -> list[str]:
    """Return list of available MATLAB model files in the package data directory.

    Returns:
        List of .mat filenames.
    """
    data_directory = resources.files(__package__) / "data"
    return [entry.name for entry in data_directory.iterdir() if entry.is_file() and entry.name.endswith(".mat")]


def load_bayesian_network_model_from_file(
    filepath: str | Path,
) -> BayesianNetworkModelData | None:
    """Load a Bayesian network model from a MATLAB file.

    Args:
        filepath: Path to the MATLAB file.

    Returns:
        BayesianNetworkModelData instance or None if loading fails.
    """
    filepath = Path(filepath)
    base_name = filepath.name

    try:
        data_context: AbstractContextManager[Path] = nullcontext(filepath)

        if not filepath.is_file():
            resource_path = resources.files(__package__) / "data" / base_name
            if not resource_path.is_file():
                print(f"The file {filepath} was not found in the current directory or in the package.")
                return None
            data_context = resources.as_file(resource_path)

        with data_context as resolved_path:
            matlab_data = scipy.io.loadmat(str(resolved_path), mat_dtype=True)

        return BayesianNetworkModelData.from_matlab_dictionary(matlab_data)

    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
        return None
    except (OSError, ValueError, KeyError) as error:
        print(f"An error occurred while loading the file: {error}")
        return None
    except Exception as error:  # noqa: BLE001
        # Catch any other errors (e.g., MatReadError from scipy)
        print(f"An error occurred while loading the file: {error}")
        return None


# =============================================================================
# Visualization
# =============================================================================


class TrackVisualizationRenderer:
    """Creates visualizations of generated aircraft tracks."""

    @staticmethod
    def render_three_dimensional_tracks(
        track_results: Sequence[TrackResultData],
        plot_title: str = "Track Generation Tool",
        output_filepath: Path | str | None = None,
        output_directory: Path | None = None,
        output_filename_base: str | None = None,
    ) -> Path | None:
        """Create 3D visualization of generated tracks.

        Args:
            track_results: Sequence of track result dictionaries.
            plot_title: Plot title.
            output_filepath: Explicit path to save the plot image. If None, uses output_directory.
            output_directory: Directory to save files in (uses DEFAULT_OUTPUT_DIRECTORY if None).
            output_filename_base: Base filename for auto-generated path (required if output_filepath is None
                and saving to file is desired).

        Returns:
            Path to saved file if saving, None if displaying interactively.
        """
        figure = plt.figure()
        axes = figure.add_subplot(111, projection="3d")

        for track_index, track_result in enumerate(track_results):
            # Support both old and new key names
            north_key = "north_position_feet" if "north_position_feet" in track_result else "north_ft"
            east_key = "east_position_feet" if "east_position_feet" in track_result else "east_ft"
            altitude_key = "altitude_feet" if "altitude_feet" in track_result else "up_ft"

            axes.plot3D(
                track_result[north_key],  # type: ignore[literal-required]
                track_result[east_key],  # type: ignore[literal-required]
                track_result[altitude_key],  # type: ignore[literal-required]
                label=f"Track {track_index + 1}",
            )

        axes.set_xlabel("North (ft)")
        axes.set_ylabel("East (ft)")
        axes.set_zlabel("Up (ft)")
        axes.set_title(plot_title)
        axes.legend()

        # Determine save path
        save_path: Path | None = None
        if output_filepath is not None:
            save_path = Path(output_filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        elif output_filename_base is not None:
            target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
            save_path = generate_unique_filepath(target_directory, f"{output_filename_base}_tracks", ".png")

        if save_path is not None:
            figure.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            return save_path

        plt.show()
        return None


# =============================================================================
# Track Generation Session - Replaces Global State
# =============================================================================


@dataclass
class TrackGenerationSession:
    """Encapsulates a track generation session, replacing global state.

    This class maintains session state and provides a clean interface
    for the track generation workflow.
    """

    model_data: BayesianNetworkModelData
    model_name: str
    generator: AircraftTrackGenerator
    generated_tracks: list[TrackResultData] = field(default_factory=list)
    initial_conditions: pd.DataFrame | None = None
    transition_data: list[list[list[float]]] = field(default_factory=list)

    @classmethod
    def create_from_file(cls, filepath: str | Path) -> TrackGenerationSession | None:
        """Create a track generation session from a model file.

        Args:
            filepath: Path to the MATLAB model file.

        Returns:
            TrackGenerationSession instance or None if loading fails.
        """
        model_data = load_bayesian_network_model_from_file(filepath)
        if model_data is None:
            return None

        model_name = Path(filepath).stem
        generator = AircraftTrackGenerator(model_data)

        return cls(
            model_data=model_data,
            model_name=model_name,
            generator=generator,
        )

    def generate_tracks(
        self,
        number_of_tracks: int,
        simulation_duration_seconds: int,
        use_reproducible_seed: bool = False,
    ) -> list[TrackResultData]:
        """Generate tracks and store results in session.

        Args:
            number_of_tracks: Number of tracks to generate.
            simulation_duration_seconds: Duration of each track in seconds.
            use_reproducible_seed: Enable seeded generation for reproducibility.

        Returns:
            List of generated track results.
        """
        self.generated_tracks, self.initial_conditions, self.transition_data = self.generator.generate_multiple_tracks(
            number_of_tracks,
            simulation_duration_seconds,
            use_reproducible_seed,
        )
        return self.generated_tracks

    def export_to_csv(self, output_filename_base: str | None = None) -> list[Path]:
        """Export generated tracks to CSV format.

        Args:
            output_filename_base: Base filename (uses model name if not provided).

        Returns:
            List of created file paths.
        """
        filename_base = output_filename_base or self.model_name
        exporter = CsvTrackResultExporter()
        return exporter.export_tracks(self.generated_tracks, filename_base)

    def export_to_matlab(self, output_filename_base: str | None = None) -> Path | None:
        """Export generated tracks to MATLAB format.

        Args:
            output_filename_base: Base filename (uses model name if not provided).

        Returns:
            Path to created file or None if no tracks.
        """
        if not self.generated_tracks:
            print("No tracks to export.")
            return None

        filename_base = output_filename_base or self.model_name
        exporter = MatlabTrackResultExporter()
        return exporter.export_tracks(self.generated_tracks, filename_base)

    def visualize_tracks(
        self,
        plot_title: str | None = None,
        output_filepath: Path | str | None = None,
        save_to_file: bool = True,
    ) -> Path | None:
        """Visualize generated tracks in 3D.

        Args:
            plot_title: Plot title (uses model name if not provided).
            output_filepath: Explicit path to save the plot image.
            save_to_file: If True and output_filepath is None, saves to output directory
                with unique filename. If False, displays interactively.

        Returns:
            Path to saved file if saving, None if displaying interactively.
        """
        title = plot_title or self.model_name.replace("_", " ")
        filename_base = self.model_name if save_to_file else None
        return TrackVisualizationRenderer.render_three_dimensional_tracks(
            self.generated_tracks,
            title,
            output_filepath=output_filepath,
            output_filename_base=filename_base,
        )


# =============================================================================
# Public API Functions - Backward Compatible Interface
# =============================================================================


def generate_aircraft_tracks(
    model_filepath: str,
    simulation_duration_seconds: int,
    number_of_tracks: int,
    use_reproducible_seed: bool = False,
) -> tuple[list[TrackResultData], TrackGenerationSession] | None:
    """Generate aircraft tracks from a Bayesian network model.

    This is the main entry point for track generation with the refactored API.

    Args:
        model_filepath: Path to MATLAB model file or model name.
        simulation_duration_seconds: Track duration in seconds.
        number_of_tracks: Number of tracks to generate.
        use_reproducible_seed: Enable seeded generation for reproducibility.

    Returns:
        Tuple of (track results, session) or None if model loading fails.
    """
    start_time = time.time()

    session = TrackGenerationSession.create_from_file(model_filepath)
    if session is None:
        return None

    results = session.generate_tracks(
        number_of_tracks,
        simulation_duration_seconds,
        use_reproducible_seed,
    )

    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    return results, session


# =============================================================================
# Legacy API - Backward Compatible Functions (Deprecated)
# =============================================================================


def gen_track(
    filename: str,
    sample_time: int,
    track_num: int,
    seed: bool = False,
) -> list[TrackResult] | None:
    """Generate aircraft tracks (legacy interface - deprecated).

    This function is provided for backward compatibility.
    Use generate_aircraft_tracks() for new code.

    Args:
        filename: Path to MATLAB model file or model name.
        sample_time: Track duration in seconds.
        track_num: Number of tracks to generate.
        seed: Enable seeded generation for reproducibility.

    Returns:
        List of valid track results, or None if file not found.
    """
    result = generate_aircraft_tracks(filename, sample_time, track_num, seed)
    if result is None:
        return None

    tracks, _session = result

    # Convert to legacy format
    legacy_tracks: list[TrackResult] = []
    for track in tracks:
        legacy_track: TrackResult = {
            "time": track["time"],
            "north_ft": track.get("north_position_feet", track.get("north_ft", np.array([]))),  # type: ignore[typeddict-item]
            "east_ft": track.get("east_position_feet", track.get("east_ft", np.array([]))),  # type: ignore[typeddict-item]
            "up_ft": track.get("altitude_feet", track.get("up_ft", np.array([]))),  # type: ignore[typeddict-item]
            "speed_ftps": track.get("speed_feet_per_second", track.get("speed_ftps", np.array([]))),  # type: ignore[typeddict-item]
            "phi_rad": track.get("bank_angle_radians", track.get("phi_rad", np.array([]))),  # type: ignore[typeddict-item]
            "theta_rad": track.get("pitch_angle_radians", track.get("theta_rad", np.array([]))),  # type: ignore[typeddict-item]
            "psi_rad": track.get("heading_angle_radians", track.get("psi_rad", np.array([]))),  # type: ignore[typeddict-item]
        }
        legacy_tracks.append(legacy_track)

    return legacy_tracks


def generate_plot(
    results: list[TrackResult],
    title: str | None = None,
) -> None:
    """Create 3D visualization of generated tracks (legacy interface).

    Args:
        results: List of track result dictionaries.
        title: Plot title.
    """
    plot_title = title or "Track Generation Tool"
    TrackVisualizationRenderer.render_three_dimensional_tracks(results, plot_title)


def save_as_matlab(
    results: list[TrackResult],
    name: str | None = None,
) -> None:
    """Save track results to MATLAB format (legacy interface).

    Args:
        results: Track results to save.
        name: Base filename.
    """
    filename_base = name or "result_model"
    exporter = MatlabTrackResultExporter()
    exporter.export_tracks(results, filename_base)


def save_to_csv(
    results: list[TrackResult],
    filename_base: str | None = None,
) -> None:
    """Export track data to CSV format (legacy interface).

    Args:
        results: Track results list.
        filename_base: Base filename for output.
    """
    base_name = filename_base or "result_model"
    exporter = CsvTrackResultExporter()
    exporter.export_tracks(results, base_name)


def get_mat_files() -> list[str]:
    """Return list of available MATLAB model files (legacy interface).

    Returns:
        List of .mat filenames.
    """
    return get_available_model_files()


# =============================================================================
# Legacy Function Aliases
# =============================================================================

get_ind = calculate_conditional_probability_table_index
get_random = InverseTransformDistributionSampler.sample_from_distribution
revert_discretization = convert_discrete_bin_to_continuous_value


def get_unique_filename(base: str, extension: str, output_directory: Path | None = None) -> str:
    """Legacy function alias for generate_unique_filepath."""
    target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
    return str(generate_unique_filepath(target_directory, base, extension))
