"""Constants and configuration values for aircraft track generation.

This module contains all constant values organized by domain,
following the Single Responsibility Principle.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final


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


# Default output directory for all generated files
DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path("output")
