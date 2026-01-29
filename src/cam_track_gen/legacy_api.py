"""Legacy API functions for backward compatibility.

This module provides the original function-based API to maintain
backward compatibility with existing code that uses the older interface.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from cam_track_gen.generator import (
    TrackGenerationSession,
    generate_aircraft_tracks,
    get_available_model_files,
    load_bayesian_network_model_from_file,
)
from cam_track_gen.visualization import TrackVisualizationRenderer

if TYPE_CHECKING:
    from cam_track_gen.types import TrackResultData


def get_mat_files() -> list[str]:
    """Get a list of available MATLAB model files.

    Returns:
        List of available model file names.
    """
    return get_available_model_files()


def gen_track(
    model_filepath: str,
    simulation_duration_seconds: int,
    number_of_tracks: int,
    seed: bool = False,
    use_reproducible_seed: bool | None = None,
) -> list[TrackResultData] | None:
    """Legacy function to generate aircraft tracks.

    This is an alias for generate_aircraft_tracks that returns only
    the track results for backward compatibility.

    Args:
        model_filepath: Path to MATLAB model file or model name.
        simulation_duration_seconds: Track duration in seconds.
        number_of_tracks: Number of tracks to generate.
        seed: Enable seeded generation for reproducibility (legacy parameter name).
        use_reproducible_seed: Enable seeded generation for reproducibility (new parameter name).

    Returns:
        List of track results or None if model loading fails.
    """
    # Support both parameter names for backward compatibility
    reproducible = use_reproducible_seed if use_reproducible_seed is not None else seed

    result = generate_aircraft_tracks(
        model_filepath,
        simulation_duration_seconds,
        number_of_tracks,
        reproducible,
    )
    if result is None:
        return None
    return result[0]


def generate_plot(
    track_data: list[TrackResultData],
    plot_title: str = "Aircraft Tracks",
    output_filepath: Path | str | None = None,
) -> None:
    """Legacy function to visualize aircraft tracks.

    Args:
        track_data: List of track result dictionaries.
        plot_title: Title for the plot.
        output_filepath: Optional path to save the plot image.
    """
    TrackVisualizationRenderer.render_three_dimensional_tracks(
        track_data,
        plot_title,
        output_filepath=output_filepath,
        output_filename_base=None,  # Don't auto-save
    )


def save_as_matlab(
    track_data: list[TrackResultData],
    output_filename_base: str,
) -> Path | None:
    """Legacy function to save tracks to MATLAB format.

    Args:
        track_data: List of track result dictionaries.
        output_filename_base: Base filename for output.

    Returns:
        Path to created file or None if no tracks.
    """
    from cam_track_gen.exporters import MatlabTrackResultExporter

    if not track_data:
        print("No tracks to export.")
        return None

    exporter = MatlabTrackResultExporter()
    return exporter.export_tracks(track_data, output_filename_base)


def save_to_csv(
    track_data: list[TrackResultData],
    output_filename_base: str,
) -> list[Path]:
    """Legacy function to save tracks to CSV format.

    Args:
        track_data: List of track result dictionaries.
        output_filename_base: Base filename for output.

    Returns:
        List of created file paths.
    """
    from cam_track_gen.exporters import CsvTrackResultExporter

    exporter = CsvTrackResultExporter()
    return exporter.export_tracks(track_data, output_filename_base)


# Expose the key classes for direct import
__all__ = [
    "gen_track",
    "generate_aircraft_tracks",
    "generate_plot",
    "get_available_model_files",
    "get_mat_files",
    "load_bayesian_network_model_from_file",
    "save_as_matlab",
    "save_to_csv",
    "TrackGenerationSession",
]
