"""Track result exporters for aircraft track generation.

This module contains classes for exporting track results
to various file formats (CSV, MATLAB).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import scipy.io

from cam_track_gen.constants import DEFAULT_OUTPUT_DIRECTORY, FileExportLimits
from cam_track_gen.types import TrackResultData, TrackResultExporterInterface
from cam_track_gen.utilities import generate_unique_filepath

if TYPE_CHECKING:
    from collections.abc import Sequence


class CsvTrackResultExporter(TrackResultExporterInterface):
    """Exports track results to CSV format with automatic file splitting."""

    def __init__(
        self,
        maximum_rows_per_file: int = FileExportLimits.MAXIMUM_CSV_ROWS_PER_FILE,
    ) -> None:
        """Initialize the CSV exporter.

        Args:
            maximum_rows_per_file: Maximum number of rows per output file.
        """
        self.maximum_rows_per_file = maximum_rows_per_file

    def export_tracks(
        self,
        track_results: Sequence[TrackResultData],
        output_filename_base: str,
        output_directory: Path | None = None,
    ) -> list[Path]:
        """Export track data to CSV format with automatic file splitting.

        Args:
            track_results: Sequence of track results to export.
            output_filename_base: Base filename for output.
            output_directory: Directory to save files in (uses DEFAULT_OUTPUT_DIRECTORY if None).

        Returns:
            List of paths to created files.
        """
        if not track_results:
            print("No track data to export.")
            return []

        target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
        created_files: list[Path] = []
        fieldnames = ["Aircraft_ID"] + list(track_results[0].keys())
        current_row_count = 0

        output_filepath = generate_unique_filepath(target_directory, f"{output_filename_base}_Result", ".csv")
        current_file = open(output_filepath, mode="w", newline="", encoding="utf-8")  # noqa: SIM115
        csv_writer = csv.writer(current_file)
        csv_writer.writerow(fieldnames)
        created_files.append(output_filepath)
        print(f"Writing to {output_filepath}...")

        try:
            for aircraft_id, track in enumerate(track_results, start=1):
                track_length = len(track["time"])
                for row_index in range(track_length):
                    if current_row_count >= self.maximum_rows_per_file:
                        current_file.close()
                        output_filepath = generate_unique_filepath(
                            target_directory, f"{output_filename_base}_Result", ".csv"
                        )
                        current_file = open(output_filepath, mode="w", newline="", encoding="utf-8")  # noqa: SIM115
                        csv_writer = csv.writer(current_file)
                        csv_writer.writerow(fieldnames)
                        created_files.append(output_filepath)
                        print(f"Writing to {output_filepath}...")
                        current_row_count = 0

                    row = [aircraft_id] + [track[key][row_index] for key in track]  # type: ignore[literal-required]
                    csv_writer.writerow(row)
                    current_row_count += 1
        finally:
            current_file.close()

        print("Data successfully saved.")
        return created_files


class MatlabTrackResultExporter(TrackResultExporterInterface):
    """Exports track results to MATLAB .mat format."""

    def export_tracks(
        self,
        track_results: Sequence[TrackResultData],
        output_filename_base: str,
        output_directory: Path | None = None,
    ) -> Path:
        """Export track results to MATLAB format.

        Args:
            track_results: Sequence of track results to export.
            output_filename_base: Base filename for output.
            output_directory: Directory to save files in (uses DEFAULT_OUTPUT_DIRECTORY if None).

        Returns:
            Path to created file.
        """
        target_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
        output_filepath = generate_unique_filepath(target_directory, f"{output_filename_base}_Result", ".mat")

        scipy.io.savemat(str(output_filepath), {"results": track_results})
        print(f"Data successfully saved to {output_filepath}")
        return output_filepath
