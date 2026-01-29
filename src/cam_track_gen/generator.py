"""Main track generator and session management.

This module contains the main track generator class and session management
for the aircraft track generation workflow.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.io
from numpy.typing import NDArray

from cam_track_gen.bayesian_network import (
    BayesianNetworkStateSampler,
    PerformanceLimitsCalculator,
    SampledDataToTrackConverter,
)
from cam_track_gen.constants import CutPointLabels
from cam_track_gen.data_classes import (
    BayesianNetworkModelData,
    TrackGenerationConfiguration,
)
from cam_track_gen.exporters import CsvTrackResultExporter, MatlabTrackResultExporter
from cam_track_gen.types import TrackResultData, TrackValidatorInterface
from cam_track_gen.validation import ConstraintBasedTrackValidator
from cam_track_gen.visualization import TrackVisualizationRenderer

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class AircraftTrackGenerator:
    """Main class for generating aircraft tracks from Bayesian network models.

    This class orchestrates the track generation process using dependency injection
    for all components, following the Dependency Inversion Principle.
    """

    def __init__(
        self,
        model_data: BayesianNetworkModelData,
        sampler: BayesianNetworkStateSampler | None = None,
        validator: TrackValidatorInterface | None = None,
    ) -> None:
        """Initialize the track generator.

        Args:
            model_data: Loaded Bayesian network model data.
            sampler: Optional custom sampler (created if not provided).
            validator: Optional custom validator (created if not provided).
        """
        self.model_data = model_data
        self.sampler = sampler or BayesianNetworkStateSampler(model_data)
        self._configuration = self._initialize_configuration()
        self._data_converter = SampledDataToTrackConverter(self._configuration)
        self.validator = validator or ConstraintBasedTrackValidator(
            self._configuration.altitude_boundary,
            self._configuration.performance_limits,
        )

    def _initialize_configuration(self) -> TrackGenerationConfiguration:
        """Initialize the track generation configuration."""
        number_of_initial_variables = len(self.model_data.variable_labels)

        cut_points_labels = self._get_cut_points_labels(number_of_initial_variables)
        cut_points_order = self._compute_cut_points_order(cut_points_labels)

        acceleration_label = (
            CutPointLabels.ACCELERATION_LEGACY
            if CutPointLabels.ACCELERATION_LEGACY in cut_points_labels
            else CutPointLabels.ACCELERATION
        )
        acceleration_index = cut_points_labels.index(acceleration_label)
        turn_rate_index = cut_points_labels.index(CutPointLabels.TURN_RATE)

        discretization_boundaries = self.model_data.discretization_cut_points[cut_points_order, 1]
        limits_calculator = PerformanceLimitsCalculator(self.model_data)
        altitude_boundary, performance_limits = limits_calculator.calculate_limits(
            discretization_boundaries, is_rotorcraft=False
        )

        resample_probabilities = self.model_data.resample_rate_matrix[-3:, 0]

        return TrackGenerationConfiguration(
            model_data=self.model_data,
            cut_points_variable_order=cut_points_order,
            acceleration_variable_index=acceleration_index,
            turn_rate_variable_index=turn_rate_index,
            resample_probabilities=resample_probabilities,
            altitude_boundary=altitude_boundary,
            performance_limits=performance_limits,
            initial_topological_sort_order=self.sampler.initial_topological_order,
            transition_topological_sort_order=self.sampler.transition_topological_order,
        )

    def _get_cut_points_labels(self, number_of_variables: int) -> list[str]:
        """Get cut points labels based on number of variables."""
        if number_of_variables == 6:
            # Handle legacy typo in some data files
            if self.model_data.discretization_cut_points[0, 0][0] == CutPointLabels.ACCELERATION_LEGACY:
                return [
                    CutPointLabels.AIRSPACE,
                    CutPointLabels.ALTITUDE,
                    CutPointLabels.SPEED,
                    CutPointLabels.ACCELERATION_LEGACY,
                    CutPointLabels.VERTICAL_RATE,
                    CutPointLabels.TURN_RATE,
                ]
            return [
                CutPointLabels.AIRSPACE,
                CutPointLabels.ALTITUDE,
                CutPointLabels.SPEED,
                CutPointLabels.ACCELERATION,
                CutPointLabels.VERTICAL_RATE,
                CutPointLabels.TURN_RATE,
            ]
        return [
            CutPointLabels.WEIGHT_TURBULENCE_CATEGORY,
            CutPointLabels.AIRSPACE,
            CutPointLabels.ALTITUDE,
            CutPointLabels.SPEED,
            CutPointLabels.ACCELERATION,
            CutPointLabels.VERTICAL_RATE,
            CutPointLabels.TURN_RATE,
        ]

    def _compute_cut_points_order(self, cut_points_labels: list[str]) -> list[int]:
        """Compute the order of cut points based on labels."""
        return [
            int(np.nonzero(self.model_data.discretization_cut_points[:, 0] == label)[0][0])
            for label in cut_points_labels
        ]

    def generate_multiple_tracks(
        self,
        number_of_tracks: int,
        simulation_duration_seconds: int,
        use_reproducible_seed: bool = False,
    ) -> tuple[list[TrackResultData], pd.DataFrame, list[list[list[float]]]]:
        """Generate multiple valid aircraft tracks.

        Args:
            number_of_tracks: Number of tracks to generate.
            simulation_duration_seconds: Duration of each track in seconds.
            use_reproducible_seed: Enable seeded generation for reproducibility.

        Returns:
            Tuple of (valid_track_results, initial_conditions_dataframe, transition_data_list).
        """
        valid_track_results: list[TrackResultData] = []
        all_initial_conditions: list[NDArray[np.int_]] = []
        all_transition_sequences: list[list[list[float]]] = []

        seed_counter = 1 if use_reproducible_seed else None
        total_valid_tracks = 0

        while total_valid_tracks < number_of_tracks:
            if use_reproducible_seed:
                np.random.seed(seed_counter)
                seed_counter = (seed_counter or 0) + 1
            else:
                np.random.seed(None)

            # Sample initial conditions
            initial_conditions = self.sampler.sample_initial_state_conditions()

            # Sample transition sequence
            transition_sequence = self.sampler.sample_transition_state_sequence(
                initial_conditions, simulation_duration_seconds
            )

            # Convert to dictionary format
            initial_conditions_dict = {
                label: int(initial_conditions[index])
                for index, label in enumerate(self.model_data.variable_labels)
            }

            # Generate track
            track_result = self._data_converter.convert_sampled_data_to_track(
                initial_conditions_dict,
                transition_sequence,
                simulation_duration_seconds,
            )

            # Validate track
            if self.validator.validate_track(track_result):
                valid_track_results.append(track_result)
                all_initial_conditions.append(initial_conditions)
                all_transition_sequences.append(transition_sequence)
                total_valid_tracks += 1

        # Create DataFrame from initial conditions
        initial_conditions_dataframe = pd.DataFrame(
            all_initial_conditions,
            columns=self.model_data.variable_labels,
        )

        return valid_track_results, initial_conditions_dataframe, all_transition_sequences

    @property
    def configuration(self) -> TrackGenerationConfiguration:
        """Get the track generation configuration."""
        return self._configuration


def get_available_model_files() -> list[str]:
    """Return list of available MATLAB model files in the package data directory.

    Returns:
        List of .mat filenames.
    """
    data_directory = resources.files(__package__) / "data"
    return [
        entry.name for entry in data_directory.iterdir() if entry.is_file() and entry.name.endswith(".mat")
    ]


def load_bayesian_network_model_from_file(
    filepath: str | Path,
) -> BayesianNetworkModelData | None:
    """Load a Bayesian network model from a MATLAB file.

    Args:
        filepath: Path to the MATLAB file.

    Returns:
        BayesianNetworkModelData instance or None if loading fails.
    """
    filepath = Path(filepath)
    base_name = filepath.name

    try:
        data_context: AbstractContextManager[Path] = nullcontext(filepath)

        if not filepath.is_file():
            resource_path = resources.files(__package__) / "data" / base_name
            if not resource_path.is_file():
                print(f"The file {filepath} was not found in the current directory or in the package.")
                return None
            data_context = resources.as_file(resource_path)

        with data_context as resolved_path:
            matlab_data = scipy.io.loadmat(str(resolved_path), mat_dtype=True)

        return BayesianNetworkModelData.from_matlab_dictionary(matlab_data)

    except FileNotFoundError:
        print(f"The file {filepath} was not found.")
        return None
    except (OSError, ValueError, KeyError) as error:
        print(f"An error occurred while loading the file: {error}")
        return None
    except Exception as error:  # noqa: BLE001
        # Catch any other errors (e.g., MatReadError from scipy)
        print(f"An error occurred while loading the file: {error}")
        return None


@dataclass
class TrackGenerationSession:
    """Encapsulates a track generation session, replacing global state.

    This class maintains session state and provides a clean interface
    for the track generation workflow.
    """

    model_data: BayesianNetworkModelData
    model_name: str
    generator: AircraftTrackGenerator
    generated_tracks: list[TrackResultData] = field(default_factory=list)
    initial_conditions: pd.DataFrame | None = None
    transition_data: list[list[list[float]]] = field(default_factory=list)

    @classmethod
    def create_from_file(cls, filepath: str | Path) -> TrackGenerationSession | None:
        """Create a track generation session from a model file.

        Args:
            filepath: Path to the MATLAB model file.

        Returns:
            TrackGenerationSession instance or None if loading fails.
        """
        model_data = load_bayesian_network_model_from_file(filepath)
        if model_data is None:
            return None

        model_name = Path(filepath).stem
        generator = AircraftTrackGenerator(model_data)

        return cls(
            model_data=model_data,
            model_name=model_name,
            generator=generator,
        )

    def generate_tracks(
        self,
        number_of_tracks: int,
        simulation_duration_seconds: int,
        use_reproducible_seed: bool = False,
    ) -> list[TrackResultData]:
        """Generate tracks and store results in session.

        Args:
            number_of_tracks: Number of tracks to generate.
            simulation_duration_seconds: Duration of each track in seconds.
            use_reproducible_seed: Enable seeded generation for reproducibility.

        Returns:
            List of generated track results.
        """
        self.generated_tracks, self.initial_conditions, self.transition_data = (
            self.generator.generate_multiple_tracks(
                number_of_tracks,
                simulation_duration_seconds,
                use_reproducible_seed,
            )
        )
        return self.generated_tracks

    def export_to_csv(self, output_filename_base: str | None = None) -> list[Path]:
        """Export generated tracks to CSV format.

        Args:
            output_filename_base: Base filename (uses model name if not provided).

        Returns:
            List of created file paths.
        """
        filename_base = output_filename_base or self.model_name
        exporter = CsvTrackResultExporter()
        return exporter.export_tracks(self.generated_tracks, filename_base)

    def export_to_matlab(self, output_filename_base: str | None = None) -> Path | None:
        """Export generated tracks to MATLAB format.

        Args:
            output_filename_base: Base filename (uses model name if not provided).

        Returns:
            Path to created file or None if no tracks.
        """
        if not self.generated_tracks:
            print("No tracks to export.")
            return None

        filename_base = output_filename_base or self.model_name
        exporter = MatlabTrackResultExporter()
        return exporter.export_tracks(self.generated_tracks, filename_base)

    def visualize_tracks(
        self,
        plot_title: str | None = None,
        output_filepath: Path | str | None = None,
        save_to_file: bool = True,
    ) -> Path | None:
        """Visualize generated tracks in 3D.

        Args:
            plot_title: Plot title (uses model name if not provided).
            output_filepath: Explicit path to save the plot image.
            save_to_file: If True and output_filepath is None, saves to output directory
                with unique filename. If False, displays interactively.

        Returns:
            Path to saved file if saving, None if displaying interactively.
        """
        title = plot_title or self.model_name.replace("_", " ")
        filename_base = self.model_name if save_to_file else None
        return TrackVisualizationRenderer.render_three_dimensional_tracks(
            self.generated_tracks,
            title,
            output_filepath=output_filepath,
            output_filename_base=filename_base,
        )


def generate_aircraft_tracks(
    model_filepath: str,
    simulation_duration_seconds: int,
    number_of_tracks: int,
    use_reproducible_seed: bool = False,
) -> tuple[list[TrackResultData], TrackGenerationSession] | None:
    """Generate aircraft tracks from a Bayesian network model.

    This is the main entry point for track generation with the refactored API.

    Args:
        model_filepath: Path to MATLAB model file or model name.
        simulation_duration_seconds: Track duration in seconds.
        number_of_tracks: Number of tracks to generate.
        use_reproducible_seed: Enable seeded generation for reproducibility.

    Returns:
        Tuple of (track results, session) or None if model loading fails.
    """
    start_time = time.time()

    session = TrackGenerationSession.create_from_file(model_filepath)
    if session is None:
        return None

    results = session.generate_tracks(
        number_of_tracks,
        simulation_duration_seconds,
        use_reproducible_seed,
    )

    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    return results, session
