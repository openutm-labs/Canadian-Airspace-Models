"""Type definitions and enumerations for aircraft track generation.

This module contains type definitions, enumerations, and protocols
that define the interfaces used throughout the package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


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
            output_directory: Directory to save files in.

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
