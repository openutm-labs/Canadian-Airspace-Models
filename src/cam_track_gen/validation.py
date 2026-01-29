"""Track validation for aircraft track generation.

This module contains classes for validating generated tracks
against performance constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from cam_track_gen.data_classes import AltitudeBoundary, DynamicPerformanceLimits
from cam_track_gen.types import TrackResultData, TrackValidatorInterface


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
        altitudes: NDArray[np.floating[Any]] = track_result[altitude_key]  # type: ignore[literal-required]
        return bool(
            np.any(
                (altitudes < self.altitude_boundary.minimum_altitude_feet)
                | (altitudes > self.altitude_boundary.maximum_altitude_feet + self.altitude_boundary.validation_margin_feet)
            )
        )

    def _check_velocity_constraint_violation(self, track_result: TrackResultData) -> bool:
        """Check for velocity constraint violations."""
        speed_key = "speed_feet_per_second" if "speed_feet_per_second" in track_result else "speed_ftps"
        speeds: NDArray[np.floating[Any]] = track_result[speed_key]  # type: ignore[literal-required]
        return bool(
            np.any(
                (speeds < self.performance_limits.velocity_limits.minimum_feet_per_second)
                | (speeds > self.performance_limits.velocity_limits.maximum_feet_per_second)
            )
        )

    def _check_vertical_rate_constraint_violation(self, track_result: TrackResultData) -> bool:
        """Check for vertical rate constraint violations."""
        altitude_key = "altitude_feet" if "altitude_feet" in track_result else "up_ft"
        altitudes: NDArray[np.floating[Any]] = track_result[altitude_key]  # type: ignore[literal-required]
        times = track_result["time"]

        vertical_rate_gradient = np.gradient(altitudes, times)
        vertical_rate_magnitude = np.abs(vertical_rate_gradient)
        return bool(np.any(vertical_rate_magnitude > self.performance_limits.vertical_rate_limits.maximum_magnitude_feet_per_second))
