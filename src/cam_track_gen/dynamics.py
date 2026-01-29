"""Aircraft dynamics calculations and integration.

This module contains classes for calculating and integrating
aircraft dynamics using physics-based equations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from cam_track_gen.constants import AircraftPerformanceLimits, PhysicsConstants
from cam_track_gen.data_classes import (
    AircraftKinematicState,
    SimulationControlParameters,
    TrigonometricStateValues,
)
from cam_track_gen.types import TrackResultData
from cam_track_gen.utilities import saturate_value_within_limits


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
