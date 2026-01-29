"""Bayesian network sampling for aircraft track generation.

This module contains classes for sampling from Bayesian network
models to generate initial conditions and state transitions.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from cam_track_gen.constants import (
    AircraftPerformanceLimits,
    CutPointLabels,
    PhysicsConstants,
    StatisticalThresholds,
    UnitConversionConstants,
)
from cam_track_gen.data_classes import (
    AircraftKinematicState,
    AltitudeBoundary,
    BayesianNetworkModelData,
    DynamicPerformanceLimits,
    SimulationControlParameters,
    TrackGenerationConfiguration,
    VelocityLimits,
    VerticalRateLimits,
)
from cam_track_gen.dynamics import AircraftTrackSimulator
from cam_track_gen.types import ProbabilityDistributionSampler, TrackResultData
from cam_track_gen.utilities import (
    InverseTransformDistributionSampler,
    calculate_conditional_probability_table_index,
    convert_discrete_bin_to_continuous_value,
)


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

        self.initial_state_graph = nx.from_numpy_array(
            model_data.initial_state_directed_acyclic_graph, create_using=nx.DiGraph
        )
        self.transition_graph = nx.from_numpy_array(
            model_data.transition_directed_acyclic_graph, create_using=nx.DiGraph
        )

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

        parent_sizes = np.array(
            [array[0].shape[0] for array in self.model_data.initial_probability_tables]
        )

        for variable_index in self.initial_topological_order:
            parent_indicator_mask = self.model_data.initial_state_directed_acyclic_graph[:, variable_index]

            if np.any(parent_indicator_mask):
                conditional_index = calculate_conditional_probability_table_index(
                    parent_sizes[parent_indicator_mask == 1].tolist(),
                    sampled_variable_values[parent_indicator_mask == 1].tolist(),
                )
            else:
                conditional_index = 1

            probability_table = self.model_data.initial_probability_tables[variable_index][0]
            sampled_variable_values[variable_index] = (
                self.sampler.sample_from_distribution(probability_table[:, conditional_index - 1]) + 1
            )

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
            self.model_data.transition_directed_acyclic_graph[
                np.ix_(self.dynamic_variable_indices, self.dynamic_variable_indices)
            ]
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
                evidence_state_array[variable_index] = (
                    self.sampler.sample_from_distribution(probability_table[:, conditional_index - 1]) + 1
                )
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
            cumulative_probability_sums[variable_index] = np.cumsum(
                probability_table[:, int(conditional_index) - 1]
            )
            random_threshold_values[:, variable_index] = (
                cumulative_probability_sums[variable_index][-1] * np.random.rand(sequence_length)
            )

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
        parent_sizes = np.array(
            [array[0].shape[0] for array in self.model_data.transition_probability_tables]
        )
        parent_sizes[:number_of_initial_variables] = [
            array[0].shape[0] for array in self.model_data.initial_probability_tables
        ]
        return parent_sizes


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
        initial_continuous_values = self._convert_initial_conditions_to_continuous(
            initial_state_conditions, variable_labels
        )

        # Process transition data with resampling
        control_command_sequence = self._process_transition_data_with_resampling(
            initial_continuous_values,
            transition_sequence_data,
            simulation_duration_seconds,
            increment_offset,
        )

        # Create initial aircraft state
        initial_kinematic_state = self._create_initial_kinematic_state(
            initial_continuous_values, variable_labels
        )

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
                self.model_data.discretization_cut_points[
                    self.configuration.cut_points_variable_order[index], 1
                ]
                .flatten()
                .astype(float)
            )

            # Handle legacy scaling for acceleration and turn rate
            if index in (
                self.configuration.acceleration_variable_index,
                self.configuration.turn_rate_variable_index,
            ):
                if any(edge > 100 for edge in bin_edge_boundaries):
                    bin_edge_boundaries = bin_edge_boundaries / 100
                    self.model_data.discretization_cut_points[
                        self.configuration.cut_points_variable_order[index], 1
                    ] = bin_edge_boundaries

            discrete_bin_value = int(np.array(initial_state_conditions[label]).item())

            if np.all(np.diff(bin_edge_boundaries) == 1):
                continuous_values[index] = discrete_bin_value
            else:
                continuous_values[index] = convert_discrete_bin_to_continuous_value(
                    bin_edge_boundaries, discrete_bin_value - 1
                )

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
        velocity_feet_per_second = UnitConversionConstants.convert_knots_to_feet_per_second(
            continuous_values[variable_labels.index("Speed")]
        )
        vertical_rate_feet_per_second = UnitConversionConstants.convert_feet_per_minute_to_feet_per_second(
            continuous_values[variable_labels.index("VRate")]
        )
        heading_change_radians_per_second = UnitConversionConstants.convert_degrees_to_radians(
            continuous_values[variable_labels.index("TRate")]
        )

        # Calculate initial attitude angles
        initial_heading_radians = 0.0

        if abs(velocity_feet_per_second) < 1e-9:
            initial_pitch_radians = 0.0
        else:
            initial_pitch_radians = math.asin(vertical_rate_feet_per_second / velocity_feet_per_second)

        initial_bank_radians = math.atan(
            velocity_feet_per_second
            * heading_change_radians_per_second
            / PhysicsConstants.GRAVITATIONAL_ACCELERATION_FEET_PER_SECOND_SQUARED
        )

        from cam_track_gen.types import VariableLabel

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
        vertical_rate_cut_point_index = int(
            np.nonzero(cut_points[:, 0] == CutPointLabels.VERTICAL_RATE)[0][0]
        )

        maximum_velocity = (
            max(cut_points[speed_cut_point_index, 1].flatten())
            * UnitConversionConstants.KNOTS_TO_FEET_PER_SECOND
        )
        minimum_vertical_rate = (
            min(cut_points[vertical_rate_cut_point_index, 1].flatten())
            * UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND
        )
        maximum_vertical_rate = (
            max(cut_points[vertical_rate_cut_point_index, 1].flatten())
            * UnitConversionConstants.FEET_PER_MINUTE_TO_FEET_PER_SECOND
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
            maximum_vertical_rate_fps = 100.0

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

        low_percentile_index = int(
            np.argmax(cumulative_probability >= StatisticalThresholds.LOW_PERCENTILE)
        )
        high_percentile_index = int(
            np.argmax(cumulative_probability >= StatisticalThresholds.HIGH_PERCENTILE)
        )

        flat_boundaries = (
            bin_edge_boundaries.flatten() if bin_edge_boundaries.ndim > 1 else bin_edge_boundaries
        )

        return (
            float(flat_boundaries[low_percentile_index + 1]),
            float(flat_boundaries[high_percentile_index + 1]),
        )
