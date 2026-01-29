"""Data classes for aircraft track generation.

This module contains dataclasses that represent state, limits,
and configuration objects used throughout the package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cam_track_gen.constants import CutPointLabels, UnitConversionConstants


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
        """Create TrigonometricStateValues from Euler angles.

        Uses math module for scalar trig operations (faster than numpy for scalars).
        """
        # math module is significantly faster than numpy for scalar operations
        sin_pitch = math.sin(pitch_radians)
        cos_pitch = math.cos(pitch_radians)
        return cls(
            sine_pitch=sin_pitch,
            cosine_pitch=cos_pitch,
            tangent_pitch=sin_pitch / cos_pitch if cos_pitch != 0 else math.tan(pitch_radians),
            sine_bank=math.sin(bank_radians),
            cosine_bank=math.cos(bank_radians),
            sine_heading=math.sin(heading_radians),
            cosine_heading=math.cos(heading_radians),
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
        airspace_exists = any(item[0][0] == CutPointLabels.AIRSPACE for item in cut_points)

        if airspace_exists:
            return cut_points

        airspace_entry = [
            np.array([CutPointLabels.AIRSPACE], dtype="<U8"),
            np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]),
        ]
        cut_points_list = cut_points.tolist()
        cut_points_list.append(airspace_entry)

        augmented_cut_points = np.empty((len(cut_points_list), 2), dtype=object)
        for index, item in enumerate(cut_points_list):
            augmented_cut_points[index, 0] = item[0]
            augmented_cut_points[index, 1] = item[1]

        return augmented_cut_points


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
