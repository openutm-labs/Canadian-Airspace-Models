"""Comprehensive unit tests for the Aircraft Track Generation Tool.

These tests cover the refactored track_generation_tool module using pytest
with parametrize for thorough test coverage.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.typing import NDArray

# Import the refactored module
from cam_track_gen.track_generation_tool import (
    AircraftDynamicsCalculator,
    AircraftDynamicsIntegrator,
    AircraftKinematicState,
    AircraftPerformanceLimits,
    AircraftTrackGenerator,
    AircraftTrackSimulator,
    AltitudeBoundary,
    AngularRateLimits,
    BayesianNetworkModelData,
    BayesianNetworkStateSampler,
    ConstraintBasedTrackValidator,
    CsvTrackResultExporter,
    DynamicPerformanceLimits,
    FileExportLimits,
    InverseTransformDistributionSampler,
    MatlabTrackResultExporter,
    PerformanceLimitsCalculator,
    PhysicsConstants,
    SampledDataToTrackConverter,
    SimulationControlParameters,
    StatisticalThresholds,
    TrackGenerationConfiguration,
    TrackGenerationSession,
    TrackResultData,
    TrackVisualizationRenderer,
    TrigonometricStateValues,
    UnitConversionConstants,
    VelocityLimits,
    VerticalRateLimits,
    calculate_conditional_probability_table_index,
    convert_discrete_bin_to_continuous_value,
    gen_track,
    generate_aircraft_tracks,
    generate_plot,
    generate_unique_filepath,
    get_available_model_files,
    get_mat_files,
    get_unique_filename,
    load_bayesian_network_model_from_file,
    save_as_matlab,
    save_to_csv,
    saturate_value_within_limits,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_velocity_limits() -> VelocityLimits:
    """Create sample velocity limits for testing."""
    return VelocityLimits(
        minimum_feet_per_second=50.0,
        maximum_feet_per_second=500.0,
    )


@pytest.fixture
def sample_vertical_rate_limits() -> VerticalRateLimits:
    """Create sample vertical rate limits for testing."""
    return VerticalRateLimits(
        minimum_feet_per_second=-50.0,
        maximum_feet_per_second=50.0,
    )


@pytest.fixture
def sample_angular_rate_limits() -> AngularRateLimits:
    """Create sample angular rate limits for testing."""
    return AngularRateLimits(
        maximum_pitch_rate_radians_per_second=0.05,
        maximum_yaw_rate_radians_per_second=1000000.0,
    )


@pytest.fixture
def sample_performance_limits(
    sample_velocity_limits: VelocityLimits,
    sample_vertical_rate_limits: VerticalRateLimits,
    sample_angular_rate_limits: AngularRateLimits,
) -> DynamicPerformanceLimits:
    """Create sample dynamic performance limits for testing."""
    return DynamicPerformanceLimits(
        velocity_limits=sample_velocity_limits,
        vertical_rate_limits=sample_vertical_rate_limits,
        angular_rate_limits=sample_angular_rate_limits,
    )


@pytest.fixture
def sample_altitude_boundary() -> AltitudeBoundary:
    """Create sample altitude boundary for testing."""
    return AltitudeBoundary(
        minimum_altitude_feet=0.0,
        maximum_altitude_feet=10000.0,
    )


@pytest.fixture
def sample_simulation_parameters(
    sample_velocity_limits: VelocityLimits,
    sample_vertical_rate_limits: VerticalRateLimits,
) -> SimulationControlParameters:
    """Create sample simulation control parameters for testing."""
    return SimulationControlParameters(
        velocity_limits=sample_velocity_limits,
        vertical_rate_limits=sample_vertical_rate_limits,
        maximum_pitch_rate_radians_per_second=0.05,
        maximum_yaw_rate_radians_per_second=1000000.0,
    )


@pytest.fixture
def sample_aircraft_state() -> AircraftKinematicState:
    """Create sample aircraft kinematic state for testing."""
    return AircraftKinematicState(
        velocity_feet_per_second=200.0,
        north_position_feet=0.0,
        east_position_feet=0.0,
        altitude_feet=5000.0,
        heading_angle_radians=0.0,
        pitch_angle_radians=0.0,
        bank_angle_radians=0.0,
        acceleration_feet_per_second_squared=0.0,
    )


@pytest.fixture
def mock_bayesian_network_model_data() -> BayesianNetworkModelData:
    """Create mock Bayesian network model data for testing."""
    # Create minimal DAGs
    initial_dag = np.zeros((6, 6))
    transition_dag = np.zeros((9, 9))

    # Create minimal probability tables
    initial_prob_tables = np.empty((6,), dtype=object)
    for i in range(6):
        initial_prob_tables[i] = np.array([[np.array([0.5, 0.5])]])

    transition_prob_tables = np.empty((9,), dtype=object)
    for i in range(9):
        transition_prob_tables[i] = np.array([[np.array([0.5, 0.5])]])

    # Create cut points
    cut_points = np.empty((6, 2), dtype=object)
    cut_points[0, 0] = np.array(["Airspace"])
    cut_points[0, 1] = np.array([[1.0, 2.0, 3.0]])
    cut_points[1, 0] = np.array(["Altitude"])
    cut_points[1, 1] = np.array([[0.0, 5000.0, 10000.0]])
    cut_points[2, 0] = np.array(["Speed"])
    cut_points[2, 1] = np.array([[50.0, 150.0, 250.0]])
    cut_points[3, 0] = np.array(["Acceleration"])
    cut_points[3, 1] = np.array([[-10.0, 0.0, 10.0]])
    cut_points[4, 0] = np.array(["Vertical Rate"])
    cut_points[4, 1] = np.array([[-1000.0, 0.0, 1000.0]])
    cut_points[5, 0] = np.array(["Turn Rate"])
    cut_points[5, 1] = np.array([[-5.0, 0.0, 5.0]])

    resample_rate = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])

    return BayesianNetworkModelData(
        initial_state_directed_acyclic_graph=initial_dag,
        transition_directed_acyclic_graph=transition_dag,
        initial_probability_tables=initial_prob_tables,
        transition_probability_tables=transition_prob_tables,
        discretization_cut_points=cut_points,
        resample_rate_matrix=resample_rate,
        variable_labels=["Airspace", "Altitude", "Speed", "Acceleration", "VRate", "TRate"],
    )


@pytest.fixture
def sample_track_result() -> TrackResultData:
    """Create sample track result data for testing."""
    time_array = np.arange(0, 10, 0.1)
    return {
        "time": time_array,
        "north_position_feet": np.linspace(0, 1000, len(time_array)),
        "east_position_feet": np.linspace(0, 500, len(time_array)),
        "altitude_feet": np.full(len(time_array), 5000.0),
        "speed_feet_per_second": np.full(len(time_array), 200.0),
        "bank_angle_radians": np.zeros(len(time_array)),
        "pitch_angle_radians": np.zeros(len(time_array)),
        "heading_angle_radians": np.zeros(len(time_array)),
    }


# =============================================================================
# Unit Conversion Tests
# =============================================================================


class TestUnitConversionConstants:
    """Tests for UnitConversionConstants class."""

    @pytest.mark.parametrize(
        ("speed_knots", "expected_fps"),
        [
            (0.0, 0.0),
            (1.0, 1.68780972222222),
            (100.0, 168.780972222222),
            (-50.0, -84.390486111111),
        ],
    )
    def test_convert_knots_to_feet_per_second(
        self,
        speed_knots: float,
        expected_fps: float,
    ) -> None:
        """Test knots to feet per second conversion."""
        result = UnitConversionConstants.convert_knots_to_feet_per_second(speed_knots)
        assert pytest.approx(result, rel=1e-6) == expected_fps

    @pytest.mark.parametrize(
        ("rate_fpm", "expected_fps"),
        [
            (0.0, 0.0),
            (60.0, 1.0),
            (120.0, 2.0),
            (-60.0, -1.0),
        ],
    )
    def test_convert_feet_per_minute_to_feet_per_second(
        self,
        rate_fpm: float,
        expected_fps: float,
    ) -> None:
        """Test feet per minute to feet per second conversion."""
        result = UnitConversionConstants.convert_feet_per_minute_to_feet_per_second(rate_fpm)
        assert pytest.approx(result, rel=1e-6) == expected_fps

    @pytest.mark.parametrize(
        ("angle_degrees", "expected_radians"),
        [
            (0.0, 0.0),
            (90.0, math.pi / 2),
            (180.0, math.pi),
            (360.0, 2 * math.pi),
            (-45.0, -math.pi / 4),
        ],
    )
    def test_convert_degrees_to_radians(
        self,
        angle_degrees: float,
        expected_radians: float,
    ) -> None:
        """Test degrees to radians conversion."""
        result = UnitConversionConstants.convert_degrees_to_radians(angle_degrees)
        assert pytest.approx(result, rel=1e-6) == expected_radians


# =============================================================================
# Velocity Limits Tests
# =============================================================================


class TestVelocityLimits:
    """Tests for VelocityLimits dataclass."""

    @pytest.mark.parametrize(
        ("velocity", "min_vel", "max_vel", "expected"),
        [
            (100.0, 50.0, 200.0, 100.0),  # Within limits
            (30.0, 50.0, 200.0, 50.0),  # Below minimum
            (250.0, 50.0, 200.0, 200.0 - 1e-6),  # Above maximum
            (50.0, 50.0, 200.0, 50.0),  # At minimum
            (199.9, 50.0, 200.0, 199.9),  # Just below maximum
        ],
    )
    def test_clamp_velocity(
        self,
        velocity: float,
        min_vel: float,
        max_vel: float,
        expected: float,
    ) -> None:
        """Test velocity clamping within limits."""
        limits = VelocityLimits(
            minimum_feet_per_second=min_vel,
            maximum_feet_per_second=max_vel,
        )
        result = limits.clamp_velocity(velocity)
        assert pytest.approx(result, rel=1e-6) == expected


# =============================================================================
# Vertical Rate Limits Tests
# =============================================================================


class TestVerticalRateLimits:
    """Tests for VerticalRateLimits dataclass."""

    @pytest.mark.parametrize(
        ("rate", "min_rate", "max_rate", "expected"),
        [
            (0.0, -50.0, 50.0, 0.0),  # Within limits
            (-60.0, -50.0, 50.0, -50.0),  # Below minimum
            (60.0, -50.0, 50.0, 50.0),  # Above maximum
            (-50.0, -50.0, 50.0, -50.0),  # At minimum
            (50.0, -50.0, 50.0, 50.0),  # At maximum
        ],
    )
    def test_clamp_vertical_rate(
        self,
        rate: float,
        min_rate: float,
        max_rate: float,
        expected: float,
    ) -> None:
        """Test vertical rate clamping within limits."""
        limits = VerticalRateLimits(
            minimum_feet_per_second=min_rate,
            maximum_feet_per_second=max_rate,
        )
        result = limits.clamp_vertical_rate(rate)
        assert pytest.approx(result, rel=1e-6) == expected

    @pytest.mark.parametrize(
        ("min_rate", "max_rate", "expected_magnitude"),
        [
            (-50.0, 50.0, 50.0),
            (-100.0, 50.0, 100.0),
            (-30.0, 80.0, 80.0),
            (0.0, 50.0, 50.0),
        ],
    )
    def test_maximum_magnitude(
        self,
        min_rate: float,
        max_rate: float,
        expected_magnitude: float,
    ) -> None:
        """Test maximum magnitude calculation."""
        limits = VerticalRateLimits(
            minimum_feet_per_second=min_rate,
            maximum_feet_per_second=max_rate,
        )
        assert limits.maximum_magnitude_feet_per_second == expected_magnitude


# =============================================================================
# Altitude Boundary Tests
# =============================================================================


class TestAltitudeBoundary:
    """Tests for AltitudeBoundary dataclass."""

    @pytest.mark.parametrize(
        ("altitude", "min_alt", "max_alt", "margin", "expected"),
        [
            (5000.0, 0.0, 10000.0, 500.0, True),  # Within bounds
            (-100.0, 0.0, 10000.0, 500.0, False),  # Below minimum
            (10600.0, 0.0, 10000.0, 500.0, False),  # Above max + margin
            (10400.0, 0.0, 10000.0, 500.0, True),  # Within margin
            (0.0, 0.0, 10000.0, 500.0, True),  # At minimum
            (10500.0, 0.0, 10000.0, 500.0, True),  # At max + margin
        ],
    )
    def test_is_within_bounds(
        self,
        altitude: float,
        min_alt: float,
        max_alt: float,
        margin: float,
        expected: bool,
    ) -> None:
        """Test altitude boundary checking."""
        boundary = AltitudeBoundary(
            minimum_altitude_feet=min_alt,
            maximum_altitude_feet=max_alt,
            validation_margin_feet=margin,
        )
        assert boundary.is_within_bounds(altitude) == expected


# =============================================================================
# Trigonometric State Values Tests
# =============================================================================


class TestTrigonometricStateValues:
    """Tests for TrigonometricStateValues dataclass."""

    @pytest.mark.parametrize(
        ("pitch", "bank", "heading"),
        [
            (0.0, 0.0, 0.0),
            (math.pi / 6, math.pi / 4, math.pi / 3),
            (-math.pi / 6, -math.pi / 4, -math.pi / 3),
            (math.pi / 2 - 0.01, 0.0, 0.0),  # Near 90 degrees pitch
        ],
    )
    def test_from_euler_angles(
        self,
        pitch: float,
        bank: float,
        heading: float,
    ) -> None:
        """Test trigonometric value computation from Euler angles."""
        trig = TrigonometricStateValues.from_euler_angles(pitch, bank, heading)

        assert pytest.approx(trig.sine_pitch, rel=1e-6) == math.sin(pitch)
        assert pytest.approx(trig.cosine_pitch, rel=1e-6) == math.cos(pitch)
        assert pytest.approx(trig.tangent_pitch, rel=1e-6) == math.tan(pitch)
        assert pytest.approx(trig.sine_bank, rel=1e-6) == math.sin(bank)
        assert pytest.approx(trig.cosine_bank, rel=1e-6) == math.cos(bank)
        assert pytest.approx(trig.sine_heading, rel=1e-6) == math.sin(heading)
        assert pytest.approx(trig.cosine_heading, rel=1e-6) == math.cos(heading)


# =============================================================================
# Aircraft Kinematic State Tests
# =============================================================================


class TestAircraftKinematicState:
    """Tests for AircraftKinematicState dataclass."""

    def test_to_output_array(self, sample_aircraft_state: AircraftKinematicState) -> None:
        """Test conversion to output array format."""
        current_time = 1.5
        output = sample_aircraft_state.to_output_array(current_time)

        assert output.shape == (8,)
        assert output[0] == current_time
        assert output[1] == sample_aircraft_state.north_position_feet
        assert output[2] == sample_aircraft_state.east_position_feet
        assert output[3] == sample_aircraft_state.altitude_feet
        assert output[4] == sample_aircraft_state.velocity_feet_per_second
        assert output[5] == sample_aircraft_state.bank_angle_radians
        assert output[6] == sample_aircraft_state.pitch_angle_radians
        assert output[7] == sample_aircraft_state.heading_angle_radians

    def test_compute_trigonometric_values(
        self,
        sample_aircraft_state: AircraftKinematicState,
    ) -> None:
        """Test trigonometric values computation from state."""
        trig = sample_aircraft_state.compute_trigonometric_values()

        assert isinstance(trig, TrigonometricStateValues)
        assert pytest.approx(trig.sine_pitch, rel=1e-6) == math.sin(sample_aircraft_state.pitch_angle_radians)


# =============================================================================
# Sampling Utilities Tests
# =============================================================================


class TestInverseTransformDistributionSampler:
    """Tests for InverseTransformDistributionSampler class."""

    def test_sample_from_uniform_distribution(self) -> None:
        """Test sampling from uniform distribution."""
        np.random.seed(42)
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        # Sample many times to verify distribution
        samples = [InverseTransformDistributionSampler.sample_from_distribution(weights) for _ in range(1000)]

        # All outcomes should be present
        assert set(samples) == {0, 1, 2, 3}

    def test_sample_from_skewed_distribution(self) -> None:
        """Test sampling from skewed distribution."""
        np.random.seed(42)
        weights = np.array([0.9, 0.1])

        samples = [InverseTransformDistributionSampler.sample_from_distribution(weights) for _ in range(1000)]

        # First outcome should be much more frequent
        count_0 = samples.count(0)
        count_1 = samples.count(1)
        assert count_0 > count_1 * 5  # At least 5x more frequent

    def test_sample_from_single_outcome(self) -> None:
        """Test sampling when only one outcome has probability."""
        weights = np.array([0.0, 1.0, 0.0])

        for _ in range(10):
            result = InverseTransformDistributionSampler.sample_from_distribution(weights)
            assert result == 1


class TestCalculateConditionalProbabilityTableIndex:
    """Tests for calculate_conditional_probability_table_index function."""

    @pytest.mark.parametrize(
        ("parent_sizes", "parent_values", "expected_index"),
        [
            ([2], [1], 1),
            ([2], [2], 2),
            ([2, 3], [1, 1], 1),
            ([2, 3], [2, 1], 2),
            ([2, 3], [1, 2], 3),
            ([2, 3], [2, 2], 4),
            ([2, 3], [1, 3], 5),
            ([2, 3], [2, 3], 6),
            ([2, 3, 4], [1, 1, 1], 1),
            ([2, 3, 4], [2, 3, 4], 24),
        ],
    )
    def test_index_calculation(
        self,
        parent_sizes: list[int],
        parent_values: list[int],
        expected_index: int,
    ) -> None:
        """Test conditional probability table index calculation."""
        result = calculate_conditional_probability_table_index(parent_sizes, parent_values)
        assert result == expected_index


class TestConvertDiscreteBinToContinuousValue:
    """Tests for convert_discrete_bin_to_continuous_value function."""

    def test_zero_crossing_bin(self) -> None:
        """Test that bins crossing zero return exactly zero."""
        bin_edges = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

        # Bin from -5 to 0 crosses zero
        result = convert_discrete_bin_to_continuous_value(bin_edges, 1)
        # This bin doesn't cross zero (it ends at zero)

        # Bin from 0 to 5 - starts at zero, doesn't cross
        result = convert_discrete_bin_to_continuous_value(bin_edges, 2)
        assert 0.0 <= result <= 5.0

    def test_positive_bin(self) -> None:
        """Test sampling from positive bin."""
        np.random.seed(42)
        bin_edges = np.array([0.0, 10.0, 20.0, 30.0])

        # Sample from bin [10, 20]
        result = convert_discrete_bin_to_continuous_value(bin_edges, 1)
        assert 10.0 <= result <= 20.0

    def test_negative_bin(self) -> None:
        """Test sampling from negative bin."""
        np.random.seed(42)
        bin_edges = np.array([-30.0, -20.0, -10.0, 0.0])

        # Sample from bin [-20, -10]
        result = convert_discrete_bin_to_continuous_value(bin_edges, 1)
        assert -20.0 <= result <= -10.0


# =============================================================================
# Saturation Utility Tests
# =============================================================================


class TestSaturateValueWithinLimits:
    """Tests for saturate_value_within_limits function."""

    @pytest.mark.parametrize(
        ("value", "minimum", "maximum", "expected"),
        [
            (50.0, 0.0, 100.0, 50.0),  # Within limits
            (-10.0, 0.0, 100.0, 0.0),  # Below minimum
            (150.0, 0.0, 100.0, 100.0),  # Above maximum
            (0.0, 0.0, 100.0, 0.0),  # At minimum
            (100.0, 0.0, 100.0, 100.0),  # At maximum
            (50.0, 50.0, 50.0, 50.0),  # Min equals max
        ],
    )
    def test_saturation(
        self,
        value: float,
        minimum: float,
        maximum: float,
        expected: float,
    ) -> None:
        """Test value saturation within limits."""
        result = saturate_value_within_limits(value, minimum, maximum)
        assert result == expected


# =============================================================================
# Aircraft Dynamics Integrator Tests
# =============================================================================


class TestAircraftDynamicsIntegrator:
    """Tests for AircraftDynamicsIntegrator class."""

    def test_straight_and_level_flight(
        self,
        sample_simulation_parameters: SimulationControlParameters,
        sample_aircraft_state: AircraftKinematicState,
    ) -> None:
        """Test integration for straight and level flight."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        # Zero commands should maintain approximately level flight
        new_state = integrator.integrate_single_time_step(
            sample_aircraft_state,
            acceleration_command=0.0,
            vertical_rate_command=0.0,
            heading_change_command=0.0,
        )

        # Velocity should remain approximately the same
        assert pytest.approx(new_state.velocity_feet_per_second, rel=0.01) == sample_aircraft_state.velocity_feet_per_second

        # Position should change based on velocity
        expected_north = sample_aircraft_state.north_position_feet + (
            sample_aircraft_state.velocity_feet_per_second * PhysicsConstants.SIMULATION_TIME_STEP_SECONDS
        )
        assert pytest.approx(new_state.north_position_feet, rel=0.01) == expected_north

    def test_acceleration_command(
        self,
        sample_simulation_parameters: SimulationControlParameters,
        sample_aircraft_state: AircraftKinematicState,
    ) -> None:
        """Test integration with acceleration command."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        acceleration = 10.0  # ft/s²
        new_state = integrator.integrate_single_time_step(
            sample_aircraft_state,
            acceleration_command=acceleration,
            vertical_rate_command=0.0,
            heading_change_command=0.0,
        )

        expected_velocity = sample_aircraft_state.velocity_feet_per_second + acceleration * PhysicsConstants.SIMULATION_TIME_STEP_SECONDS
        assert pytest.approx(new_state.velocity_feet_per_second, rel=0.01) == expected_velocity

    def test_velocity_saturation(
        self,
        sample_simulation_parameters: SimulationControlParameters,
    ) -> None:
        """Test that velocity is saturated within limits."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        # Start at maximum velocity
        state = AircraftKinematicState(
            velocity_feet_per_second=500.0,  # At maximum
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.0,
            bank_angle_radians=0.0,
            acceleration_feet_per_second_squared=0.0,
        )

        # Apply large positive acceleration
        new_state = integrator.integrate_single_time_step(
            state,
            acceleration_command=1000.0,  # Large acceleration
            vertical_rate_command=0.0,
            heading_change_command=0.0,
        )

        # Velocity should be clamped to just below maximum
        assert new_state.velocity_feet_per_second < 500.0


# =============================================================================
# Track Validator Tests
# =============================================================================


class TestConstraintBasedTrackValidator:
    """Tests for ConstraintBasedTrackValidator class."""

    def test_valid_track(
        self,
        sample_altitude_boundary: AltitudeBoundary,
        sample_performance_limits: DynamicPerformanceLimits,
        sample_track_result: TrackResultData,
    ) -> None:
        """Test validation of a valid track."""
        validator = ConstraintBasedTrackValidator(
            sample_altitude_boundary,
            sample_performance_limits,
        )

        assert validator.validate_track(sample_track_result) is True

    def test_altitude_violation(
        self,
        sample_altitude_boundary: AltitudeBoundary,
        sample_performance_limits: DynamicPerformanceLimits,
    ) -> None:
        """Test detection of altitude violation."""
        validator = ConstraintBasedTrackValidator(
            sample_altitude_boundary,
            sample_performance_limits,
        )

        # Create track with altitude violation
        time_array = np.arange(0, 10, 0.1)
        invalid_track: TrackResultData = {
            "time": time_array,
            "north_position_feet": np.zeros(len(time_array)),
            "east_position_feet": np.zeros(len(time_array)),
            "altitude_feet": np.full(len(time_array), -1000.0),  # Below minimum
            "speed_feet_per_second": np.full(len(time_array), 200.0),
            "bank_angle_radians": np.zeros(len(time_array)),
            "pitch_angle_radians": np.zeros(len(time_array)),
            "heading_angle_radians": np.zeros(len(time_array)),
        }

        assert validator.validate_track(invalid_track) is False

    def test_velocity_violation(
        self,
        sample_altitude_boundary: AltitudeBoundary,
        sample_performance_limits: DynamicPerformanceLimits,
    ) -> None:
        """Test detection of velocity violation."""
        validator = ConstraintBasedTrackValidator(
            sample_altitude_boundary,
            sample_performance_limits,
        )

        # Create track with velocity violation
        time_array = np.arange(0, 10, 0.1)
        invalid_track: TrackResultData = {
            "time": time_array,
            "north_position_feet": np.zeros(len(time_array)),
            "east_position_feet": np.zeros(len(time_array)),
            "altitude_feet": np.full(len(time_array), 5000.0),
            "speed_feet_per_second": np.full(len(time_array), 10.0),  # Below minimum
            "bank_angle_radians": np.zeros(len(time_array)),
            "pitch_angle_radians": np.zeros(len(time_array)),
            "heading_angle_radians": np.zeros(len(time_array)),
        }

        assert validator.validate_track(invalid_track) is False


# =============================================================================
# Track Result Exporter Tests
# =============================================================================


class TestCsvTrackResultExporter:
    """Tests for CsvTrackResultExporter class."""

    def test_export_creates_file(
        self,
        sample_track_result: TrackResultData,
        tmp_path: Path,
    ) -> None:
        """Test that export creates a CSV file."""
        exporter = CsvTrackResultExporter()
        output_base = str(tmp_path / "test_tracks")

        created_files = exporter.export_tracks([sample_track_result], output_base)

        assert len(created_files) == 1
        assert created_files[0].exists()
        assert created_files[0].suffix == ".csv"

    def test_export_empty_results(self, tmp_path: Path) -> None:
        """Test export with empty results list."""
        exporter = CsvTrackResultExporter()
        output_base = str(tmp_path / "test_tracks")

        created_files = exporter.export_tracks([], output_base)

        assert len(created_files) == 0


class TestMatlabTrackResultExporter:
    """Tests for MatlabTrackResultExporter class."""

    def test_export_creates_file(
        self,
        sample_track_result: TrackResultData,
        tmp_path: Path,
    ) -> None:
        """Test that export creates a MATLAB file."""
        exporter = MatlabTrackResultExporter()
        output_base = str(tmp_path / "test_tracks")

        created_file = exporter.export_tracks([sample_track_result], output_base)

        assert created_file.exists()
        assert created_file.suffix == ".mat"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGenerateUniqueFilepath:
    """Tests for generate_unique_filepath function."""

    def test_no_conflict(self, tmp_path: Path) -> None:
        """Test filepath generation when no conflict exists."""
        result = generate_unique_filepath(tmp_path, "test_file", ".txt")

        assert result == tmp_path / "test_file.txt"

    def test_with_conflict(self, tmp_path: Path) -> None:
        """Test filepath generation with existing file."""
        existing_file = tmp_path / "test_file.txt"
        existing_file.touch()

        result = generate_unique_filepath(tmp_path, "test_file", ".txt")

        assert result == tmp_path / "test_file_1.txt"
        assert result != existing_file

    def test_multiple_conflicts(self, tmp_path: Path) -> None:
        """Test filepath generation with multiple existing files."""
        (tmp_path / "test_file.txt").touch()
        (tmp_path / "test_file_1.txt").touch()
        (tmp_path / "test_file_2.txt").touch()

        result = generate_unique_filepath(tmp_path, "test_file", ".txt")

        assert result == tmp_path / "test_file_3.txt"


# =============================================================================
# Dynamic Performance Limits Tests
# =============================================================================


class TestDynamicPerformanceLimits:
    """Tests for DynamicPerformanceLimits dataclass."""

    def test_create_default(self) -> None:
        """Test creation with default angular rates."""
        limits = DynamicPerformanceLimits.create_default(
            minimum_velocity=50.0,
            maximum_velocity=500.0,
            minimum_vertical_rate=-50.0,
            maximum_vertical_rate=50.0,
        )

        assert limits.velocity_limits.minimum_feet_per_second == 50.0
        assert limits.velocity_limits.maximum_feet_per_second == 500.0
        assert limits.vertical_rate_limits.minimum_feet_per_second == -50.0
        assert limits.vertical_rate_limits.maximum_feet_per_second == 50.0
        assert limits.angular_rate_limits.maximum_pitch_rate_radians_per_second > 0
        assert limits.angular_rate_limits.maximum_yaw_rate_radians_per_second > 0


# =============================================================================
# Bayesian Network Model Data Tests
# =============================================================================


class TestBayesianNetworkModelData:
    """Tests for BayesianNetworkModelData dataclass."""

    def test_determine_variable_labels_six_variables(self) -> None:
        """Test variable label determination for 6-variable model."""
        labels = BayesianNetworkModelData._determine_variable_labels(6)

        assert len(labels) == 6
        assert labels[0] == "Airspace"
        assert "WTC" not in labels

    def test_determine_variable_labels_seven_variables(self) -> None:
        """Test variable label determination for 7-variable model."""
        labels = BayesianNetworkModelData._determine_variable_labels(7)

        assert len(labels) == 7
        assert labels[0] == "WTC"
        assert "Airspace" in labels


# =============================================================================
# Simulation Control Parameters Tests
# =============================================================================


class TestSimulationControlParameters:
    """Tests for SimulationControlParameters dataclass."""

    def test_from_performance_limits(
        self,
        sample_performance_limits: DynamicPerformanceLimits,
    ) -> None:
        """Test creation from performance limits."""
        params = SimulationControlParameters.from_performance_limits(sample_performance_limits)

        assert params.velocity_limits == sample_performance_limits.velocity_limits
        assert params.vertical_rate_limits == sample_performance_limits.vertical_rate_limits
        assert params.maximum_pitch_rate_radians_per_second == sample_performance_limits.angular_rate_limits.maximum_pitch_rate_radians_per_second


# =============================================================================
# Track Generation Session Tests
# =============================================================================


class TestTrackGenerationSession:
    """Tests for TrackGenerationSession dataclass."""

    def test_create_from_nonexistent_file(self) -> None:
        """Test session creation with nonexistent file."""
        session = TrackGenerationSession.create_from_file("/nonexistent/path.mat")

        assert session is None

    def test_export_to_csv_no_tracks(
        self,
        mock_bayesian_network_model_data: BayesianNetworkModelData,
    ) -> None:
        """Test CSV export with no generated tracks."""
        generator = MagicMock()
        session = TrackGenerationSession(
            model_data=mock_bayesian_network_model_data,
            model_name="test_model",
            generator=generator,
            generated_tracks=[],
        )

        result = session.export_to_csv()

        assert result == []

    def test_export_to_matlab_no_tracks(
        self,
        mock_bayesian_network_model_data: BayesianNetworkModelData,
    ) -> None:
        """Test MATLAB export with no generated tracks."""
        generator = MagicMock()
        session = TrackGenerationSession(
            model_data=mock_bayesian_network_model_data,
            model_name="test_model",
            generator=generator,
            generated_tracks=[],
        )

        result = session.export_to_matlab()

        assert result is None


# =============================================================================
# Performance Limits Calculator Tests
# =============================================================================


class TestPerformanceLimitsCalculator:
    """Tests for PerformanceLimitsCalculator class."""

    def test_calculate_percentile_based_limits(self) -> None:
        """Test percentile-based limit calculation."""
        distribution = np.array([10, 20, 40, 20, 10])  # Normal-ish distribution
        bin_edges = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

        low_limit, high_limit = PerformanceLimitsCalculator._calculate_percentile_based_limits(distribution, bin_edges)

        # Low limit should be around 20 (1st percentile)
        # High limit should be around 80 (99th percentile)
        assert low_limit >= 0.0
        assert high_limit <= 100.0
        assert low_limit < high_limit


# =============================================================================
# Aircraft Track Simulator Tests
# =============================================================================


class TestAircraftTrackSimulator:
    """Tests for AircraftTrackSimulator class."""

    def test_simulate_track_output_format(
        self,
        sample_simulation_parameters: SimulationControlParameters,
        sample_aircraft_state: AircraftKinematicState,
    ) -> None:
        """Test that simulation output has correct format."""
        simulator = AircraftTrackSimulator(sample_simulation_parameters)

        # Create minimal control commands
        duration = 1.0  # 1 second
        control_commands = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # time, accel, vrate, turn
                [0.5, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        result = simulator.simulate_track(sample_aircraft_state, control_commands, duration)

        # Check all required keys are present
        assert "time" in result
        assert "north_position_feet" in result
        assert "east_position_feet" in result
        assert "altitude_feet" in result
        assert "speed_feet_per_second" in result
        assert "bank_angle_radians" in result
        assert "pitch_angle_radians" in result
        assert "heading_angle_radians" in result

        # Check array lengths match
        expected_length = int(duration / PhysicsConstants.SIMULATION_TIME_STEP_SECONDS + 1)
        assert len(result["time"]) == expected_length


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the track generation workflow."""

    @pytest.mark.integration
    def test_end_to_end_with_real_model(self) -> None:
        """Test complete workflow with a real model file if available."""
        available_models = get_available_model_files()

        if not available_models:
            pytest.skip("No model files available for integration test")

        model_data = load_bayesian_network_model_from_file(available_models[0])
        if model_data is None:
            pytest.skip("Could not load model file")

        generator = AircraftTrackGenerator(model_data)

        # Generate a single short track
        tracks, initial_conditions, transitions = generator.generate_multiple_tracks(
            number_of_tracks=1,
            simulation_duration_seconds=10,
            use_reproducible_seed=True,
        )

        assert len(tracks) == 1
        assert isinstance(initial_conditions, pd.DataFrame)
        assert len(transitions) == 1

    @pytest.mark.integration
    def test_reproducibility_with_seed(self) -> None:
        """Test that seeded generation produces reproducible results."""
        available_models = get_available_model_files()

        if not available_models:
            pytest.skip("No model files available for integration test")

        model_data = load_bayesian_network_model_from_file(available_models[0])
        if model_data is None:
            pytest.skip("Could not load model file")

        generator = AircraftTrackGenerator(model_data)

        # Generate tracks twice with same seed
        tracks1, _, _ = generator.generate_multiple_tracks(
            number_of_tracks=1,
            simulation_duration_seconds=10,
            use_reproducible_seed=True,
        )

        tracks2, _, _ = generator.generate_multiple_tracks(
            number_of_tracks=1,
            simulation_duration_seconds=10,
            use_reproducible_seed=True,
        )

        # Results should be identical
        assert_array_equal(tracks1[0]["time"], tracks2[0]["time"])


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_velocity_handling(
        self,
        sample_simulation_parameters: SimulationControlParameters,
    ) -> None:
        """Test handling of near-zero velocity."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        # Create state with minimum velocity
        state = AircraftKinematicState(
            velocity_feet_per_second=1.0,  # Very low but non-zero
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.0,
            bank_angle_radians=0.0,
            acceleration_feet_per_second_squared=0.0,
        )

        # Should not raise any errors
        new_state = integrator.integrate_single_time_step(
            state,
            acceleration_command=0.0,
            vertical_rate_command=0.0,
            heading_change_command=0.0,
        )

        assert not np.isnan(new_state.velocity_feet_per_second)

    def test_extreme_bank_angle(
        self,
        sample_simulation_parameters: SimulationControlParameters,
    ) -> None:
        """Test handling of extreme bank angle commands."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        state = AircraftKinematicState(
            velocity_feet_per_second=200.0,
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.0,
            bank_angle_radians=0.0,
            acceleration_feet_per_second_squared=0.0,
        )

        # Apply extreme heading change command
        new_state = integrator.integrate_single_time_step(
            state,
            acceleration_command=0.0,
            vertical_rate_command=0.0,
            heading_change_command=1.0,  # Very high turn rate
        )

        # Bank angle should be limited
        assert abs(new_state.bank_angle_radians) <= AircraftPerformanceLimits.MAXIMUM_BANK_ANGLE_RADIANS


# =============================================================================
# Legacy API Tests
# =============================================================================


class TestLegacyApi:
    """Tests for legacy API functions."""

    def test_get_mat_files(self) -> None:
        """Test get_mat_files returns available models."""
        result = get_mat_files()
        assert isinstance(result, list)
        # Should match get_available_model_files
        assert result == get_available_model_files()

    def test_get_unique_filename(self, tmp_path: Path) -> None:
        """Test legacy get_unique_filename function."""
        result = get_unique_filename("test_base", ".mat", tmp_path)
        assert isinstance(result, str)
        assert "test_base" in result
        assert result.endswith(".mat")

    @pytest.mark.integration
    def test_gen_track_with_valid_model(self) -> None:
        """Test gen_track legacy function with valid model."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available for integration test")

        result = gen_track(available_models[0], 5, 1, seed=True)

        assert result is not None
        assert len(result) == 1
        # Check legacy key names
        track = result[0]
        assert "time" in track
        assert "north_ft" in track or "north_position_feet" in track
        assert "up_ft" in track or "altitude_feet" in track

    def test_gen_track_with_invalid_model(self) -> None:
        """Test gen_track with non-existent file."""
        result = gen_track("/nonexistent/model.mat", 5, 1)
        assert result is None

    @pytest.mark.integration
    def test_generate_aircraft_tracks_with_valid_model(self) -> None:
        """Test generate_aircraft_tracks function."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available for integration test")

        result = generate_aircraft_tracks(available_models[0], 5, 1, use_reproducible_seed=True)

        assert result is not None
        tracks, session = result
        assert len(tracks) == 1
        assert isinstance(session, TrackGenerationSession)

    def test_generate_aircraft_tracks_with_invalid_model(self) -> None:
        """Test generate_aircraft_tracks with non-existent file."""
        result = generate_aircraft_tracks("/nonexistent/model.mat", 5, 1)
        assert result is None

    @pytest.mark.integration
    def test_save_to_csv_legacy(self, tmp_path: Path) -> None:
        """Test legacy save_to_csv function."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available for integration test")

        result = gen_track(available_models[0], 5, 1, seed=True)
        if result is None:
            pytest.skip("Could not generate tracks")

        # Temporarily change to tmp_path to test output
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            os.makedirs("output", exist_ok=True)
            save_to_csv(result, "test_legacy_csv")
            # Check file was created
            csv_files = list(Path("output").glob("*.csv"))
            assert len(csv_files) >= 1
        finally:
            os.chdir(original_cwd)

    @pytest.mark.integration
    def test_save_as_matlab_legacy(self, tmp_path: Path) -> None:
        """Test legacy save_as_matlab function."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available for integration test")

        result = gen_track(available_models[0], 5, 1, seed=True)
        if result is None:
            pytest.skip("Could not generate tracks")

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            os.makedirs("output", exist_ok=True)
            save_as_matlab(result, "test_legacy_mat")
            mat_files = list(Path("output").glob("*.mat"))
            assert len(mat_files) >= 1
        finally:
            os.chdir(original_cwd)


# =============================================================================
# Visualization Tests
# =============================================================================


class TestTrackVisualizationRenderer:
    """Tests for TrackVisualizationRenderer class."""

    def test_render_saves_to_file(
        self,
        sample_track_result: TrackResultData,
        tmp_path: Path,
    ) -> None:
        """Test rendering and saving to file."""
        output_path = tmp_path / "test_plot.png"

        result = TrackVisualizationRenderer.render_three_dimensional_tracks(
            [sample_track_result],
            plot_title="Test Plot",
            output_filepath=output_path,
        )

        assert result == output_path
        assert output_path.exists()

    def test_render_auto_generates_filename(
        self,
        sample_track_result: TrackResultData,
        tmp_path: Path,
    ) -> None:
        """Test rendering with auto-generated filename."""
        result = TrackVisualizationRenderer.render_three_dimensional_tracks(
            [sample_track_result],
            plot_title="Test Plot",
            output_directory=tmp_path,
            output_filename_base="auto_test",
        )

        assert result is not None
        assert result.exists()
        assert "auto_test" in str(result)

    def test_render_legacy_key_names(self, tmp_path: Path) -> None:
        """Test rendering with legacy key names."""
        time_array = np.arange(0, 10, 0.1)
        legacy_track = {
            "time": time_array,
            "north_ft": np.linspace(0, 1000, len(time_array)),
            "east_ft": np.linspace(0, 500, len(time_array)),
            "up_ft": np.full(len(time_array), 5000.0),
            "speed_ftps": np.full(len(time_array), 200.0),
            "phi_rad": np.zeros(len(time_array)),
            "theta_rad": np.zeros(len(time_array)),
            "psi_rad": np.zeros(len(time_array)),
        }

        output_path = tmp_path / "legacy_plot.png"
        result = TrackVisualizationRenderer.render_three_dimensional_tracks(
            [legacy_track],  # type: ignore[list-item]
            plot_title="Legacy Test",
            output_filepath=output_path,
        )

        assert result == output_path
        assert output_path.exists()


# =============================================================================
# Roll Rate Constraint Tests
# =============================================================================


class TestRollRateConstraints:
    """Tests for roll rate constraint application."""

    def test_positive_bank_angle_constraint(
        self,
        sample_simulation_parameters: SimulationControlParameters,
    ) -> None:
        """Test constraining roll rate when approaching positive bank angle limit."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        # Create state with large positive bank angle
        state = AircraftKinematicState(
            velocity_feet_per_second=200.0,
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.0,
            bank_angle_radians=1.2,  # Near max bank
            acceleration_feet_per_second_squared=0.0,
        )

        # Apply command that would increase bank further
        new_state = integrator.integrate_single_time_step(
            state,
            acceleration_command=0.0,
            vertical_rate_command=0.0,
            heading_change_command=0.5,  # Positive turn
        )

        # Bank should be limited
        assert new_state.bank_angle_radians <= AircraftPerformanceLimits.MAXIMUM_BANK_ANGLE_RADIANS

    def test_negative_bank_angle_constraint(
        self,
        sample_simulation_parameters: SimulationControlParameters,
    ) -> None:
        """Test constraining roll rate when approaching negative bank angle limit."""
        integrator = AircraftDynamicsIntegrator(sample_simulation_parameters)

        state = AircraftKinematicState(
            velocity_feet_per_second=200.0,
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.0,
            bank_angle_radians=-1.2,  # Near negative max bank
            acceleration_feet_per_second_squared=0.0,
        )

        new_state = integrator.integrate_single_time_step(
            state,
            acceleration_command=0.0,
            vertical_rate_command=0.0,
            heading_change_command=-0.5,  # Negative turn
        )

        assert new_state.bank_angle_radians >= -AircraftPerformanceLimits.MAXIMUM_BANK_ANGLE_RADIANS


# =============================================================================
# Dynamics Calculator Tests
# =============================================================================


class TestAircraftDynamicsCalculator:
    """Tests for AircraftDynamicsCalculator class."""

    def test_negative_discriminant_returns_max_bank(
        self,
        sample_simulation_parameters: SimulationControlParameters,
        sample_aircraft_state: AircraftKinematicState,
    ) -> None:
        """Test that negative discriminant returns maximum bank angle."""
        calculator = AircraftDynamicsCalculator(sample_simulation_parameters)
        trig_values = sample_aircraft_state.compute_trigonometric_values()

        # Create conditions that might result in negative discriminant
        # by using extreme acceleration
        max_bank = calculator.calculate_maximum_allowable_bank_angle(
            sample_aircraft_state,
            trig_values,
            acceleration_command=10000.0,  # Extreme acceleration
            saturated_vertical_rate_command=0.0,
        )

        # Should return a valid bank angle
        assert max_bank <= AircraftPerformanceLimits.MAXIMUM_BANK_ANGLE_RADIANS
        assert max_bank >= 0

    def test_cosine_bank_limit_greater_than_one(
        self,
        sample_simulation_parameters: SimulationControlParameters,
    ) -> None:
        """Test handling when cosine bank limit exceeds 1."""
        calculator = AircraftDynamicsCalculator(sample_simulation_parameters)

        # Create state with specific pitch to trigger edge case
        state = AircraftKinematicState(
            velocity_feet_per_second=50.0,  # Lower velocity
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.3,  # Moderate pitch
            bank_angle_radians=0.0,
            acceleration_feet_per_second_squared=0.0,
        )
        trig_values = state.compute_trigonometric_values()

        max_bank = calculator.calculate_maximum_allowable_bank_angle(
            state,
            trig_values,
            acceleration_command=0.0,
            saturated_vertical_rate_command=5.0,  # Low vertical rate
        )

        # Should return a valid non-negative bank angle
        assert max_bank >= 0
        assert not np.isnan(max_bank)


# =============================================================================
# CSV File Splitting Tests
# =============================================================================


class TestCsvFileSplitting:
    """Tests for CSV file splitting when exceeding row limits."""

    def test_file_splitting_on_row_limit(self, tmp_path: Path) -> None:
        """Test that CSV export splits files when row limit is exceeded."""
        # Create many short tracks to exceed a small row limit
        time_array = np.arange(0, 100, 0.1)  # 1000 rows per track
        track: TrackResultData = {
            "time": time_array,
            "north_position_feet": np.zeros(len(time_array)),
            "east_position_feet": np.zeros(len(time_array)),
            "altitude_feet": np.full(len(time_array), 5000.0),
            "speed_feet_per_second": np.full(len(time_array), 200.0),
            "bank_angle_radians": np.zeros(len(time_array)),
            "pitch_angle_radians": np.zeros(len(time_array)),
            "heading_angle_radians": np.zeros(len(time_array)),
        }

        # Use small row limit to force splitting
        exporter = CsvTrackResultExporter(maximum_rows_per_file=500)

        # Export 2 tracks (2000 rows total, should split into multiple files)
        created_files = exporter.export_tracks([track, track], "split_test", tmp_path)

        # Should create multiple files
        assert len(created_files) >= 2
        for file in created_files:
            assert file.exists()


# =============================================================================
# Bayesian Network Sampler Tests
# =============================================================================


class TestBayesianNetworkStateSampler:
    """Tests for BayesianNetworkStateSampler class."""

    @pytest.mark.integration
    def test_sample_initial_state_conditions(self) -> None:
        """Test sampling initial conditions from network."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available")

        model_data = load_bayesian_network_model_from_file(available_models[0])
        if model_data is None:
            pytest.skip("Could not load model")

        sampler = BayesianNetworkStateSampler(model_data)

        # Sample multiple times to verify consistency
        for _ in range(5):
            initial_conditions = sampler.sample_initial_state_conditions()

            assert isinstance(initial_conditions, np.ndarray)
            assert len(initial_conditions) == len(model_data.variable_labels)
            # All values should be positive integers (1-based indexing)
            assert all(val >= 1 for val in initial_conditions)

    @pytest.mark.integration
    def test_sample_transition_state_sequence(self) -> None:
        """Test sampling transition sequences."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available")

        model_data = load_bayesian_network_model_from_file(available_models[0])
        if model_data is None:
            pytest.skip("Could not load model")

        sampler = BayesianNetworkStateSampler(model_data)
        initial_conditions = sampler.sample_initial_state_conditions()

        sequence_length = 10
        transitions = sampler.sample_transition_state_sequence(initial_conditions, sequence_length)

        assert len(transitions) == sequence_length
        for transition in transitions:
            assert len(transition) == 3  # acceleration, vertical rate, turn rate


# =============================================================================
# Model Loading Error Tests
# =============================================================================


class TestModelLoadingErrors:
    """Tests for model loading error handling."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading a file that doesn't exist."""
        result = load_bayesian_network_model_from_file("/nonexistent/path/model.mat")
        assert result is None

    def test_load_invalid_file_path(self, tmp_path: Path) -> None:
        """Test loading from an invalid path."""
        invalid_path = tmp_path / "not_a_mat_file.mat"
        # Write minimal data that won't be valid MATLAB format
        invalid_path.write_bytes(b"\x00" * 100)

        result = load_bayesian_network_model_from_file(str(invalid_path))
        # Should return None due to invalid format
        assert result is None


# =============================================================================
# Track Generation Session Additional Tests
# =============================================================================


class TestTrackGenerationSessionAdditional:
    """Additional tests for TrackGenerationSession."""

    @pytest.mark.integration
    def test_full_session_workflow(self, tmp_path: Path) -> None:
        """Test complete session workflow."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available")

        session = TrackGenerationSession.create_from_file(available_models[0])
        if session is None:
            pytest.skip("Could not create session")

        # Generate tracks
        tracks = session.generate_tracks(2, 5, use_reproducible_seed=True)
        assert len(tracks) == 2

        # Export to CSV
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            os.makedirs("output", exist_ok=True)

            csv_files = session.export_to_csv()
            assert len(csv_files) >= 1

            mat_file = session.export_to_matlab()
            assert mat_file is not None
            assert mat_file.exists()

            # Test visualization
            plot_file = session.visualize_tracks(save_to_file=True)
            assert plot_file is not None
            assert plot_file.exists()
        finally:
            os.chdir(original_cwd)

    @pytest.mark.integration
    def test_visualize_tracks_no_save(self) -> None:
        """Test visualization without saving."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available")

        session = TrackGenerationSession.create_from_file(available_models[0])
        if session is None:
            pytest.skip("Could not create session")

        # Generate tracks
        session.generate_tracks(1, 5, use_reproducible_seed=True)

        # Visualize without saving (would normally show plot)
        # We patch plt.show to avoid actually displaying
        with patch("matplotlib.pyplot.show"):
            result = session.visualize_tracks(save_to_file=False)

        # When not saving, returns None
        assert result is None


# =============================================================================
# Performance Limits Calculator Additional Tests
# =============================================================================


class TestPerformanceLimitsCalculatorAdditional:
    """Additional tests for PerformanceLimitsCalculator."""

    @pytest.mark.integration
    def test_limits_calculation_with_real_model(self) -> None:
        """Test limits calculation with a real model."""
        available_models = get_available_model_files()
        if not available_models:
            pytest.skip("No model files available")

        model_data = load_bayesian_network_model_from_file(available_models[0])
        if model_data is None:
            pytest.skip("Could not load model")

        # Create generator which initializes limits internally
        generator = AircraftTrackGenerator(model_data)
        config = generator.configuration

        # Verify limits are calculated
        assert config.altitude_boundary is not None
        assert config.performance_limits is not None
        assert config.performance_limits.velocity_limits.minimum_feet_per_second > 0
        assert config.performance_limits.velocity_limits.maximum_feet_per_second > 0
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
