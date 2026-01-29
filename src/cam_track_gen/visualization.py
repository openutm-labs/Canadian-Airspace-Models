"""Track visualization for aircraft track generation.

This module contains classes for visualizing generated aircraft tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from cam_track_gen.constants import DEFAULT_OUTPUT_DIRECTORY
from cam_track_gen.types import TrackResultData
from cam_track_gen.utilities import generate_unique_filepath

if TYPE_CHECKING:
    from collections.abc import Sequence


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
