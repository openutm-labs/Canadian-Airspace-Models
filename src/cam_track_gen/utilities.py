"""Utility functions for aircraft track generation.

This module contains utility functions for sampling, file operations,
and value manipulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from cam_track_gen.constants import DEFAULT_OUTPUT_DIRECTORY

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    # Python builtins are faster for scalar values than np.clip
    if value < minimum_limit:
        return minimum_limit
    if value > maximum_limit:
        return maximum_limit
    return value


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
        # Pure Python linear scan is faster than numpy cumsum + searchsorted for small arrays
        # Typical probability tables have 2-10 bins
        total = 0.0
        for w in probability_weights:
            total += w
        threshold = total * np.random.rand()  # noqa: NPY002

        cumulative = 0.0
        for i, w in enumerate(probability_weights):
            cumulative += w
            if cumulative > threshold:
                return i
        return len(probability_weights) - 1


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
    # Optimized version avoiding numpy overhead for small arrays
    # Most calls have 1-4 parents, so pure Python is faster
    linear_index = 0
    cumulative_product = 1

    for size, value in zip(parent_variable_sizes, parent_variable_values):
        linear_index += cumulative_product * (value - 1)
        cumulative_product *= size

    return linear_index + 1


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


# Legacy function alias
def get_unique_filename(base: str, extension: str, output_directory: Path | None = None) -> str:
    """Legacy function alias for generate_unique_filepath."""
    target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
    return str(generate_unique_filepath(target_directory, base, extension))
