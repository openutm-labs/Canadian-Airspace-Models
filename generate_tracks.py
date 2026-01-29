"""Refactored test script for Canadian Airspace Models track generation.

This script demonstrates the new API for generating aircraft tracks
using the refactored track generation tool.

All output files are automatically saved to the 'output' directory
with unique filenames to prevent overwriting.
"""

from cam_track_gen import (
    TrackGenerationSession,
    TrackVisualizationRenderer,
    get_available_model_files,
)


def main() -> None:
    """Generate aircraft tracks using the refactored session-based API.

    All files are automatically saved to the 'output' directory with
    unique filenames to prevent overwriting existing files.
    """
    # Configuration parameters
    simulation_duration_seconds = 250
    number_of_tracks = 10
    model_filename = "Light_Aircraft_Below_10000_ft_Data.mat"

    # Display available model files
    available_models = get_available_model_files()
    print(f"Available model files: {available_models}")

    # Create a track generation session from the model file
    session = TrackGenerationSession.create_from_file(model_filename)

    if session is None:
        print(f"Failed to load model file: {model_filename}")
        return

    print(f"Loaded model: {session.model_name}")

    # Generate tracks
    track_results = session.generate_tracks(
        number_of_tracks=number_of_tracks,
        simulation_duration_seconds=simulation_duration_seconds,
        use_reproducible_seed=False,
    )

    if not track_results:
        print("No tracks were generated.")
        return

    print(f"Successfully generated {len(track_results)} tracks")

    # Export results to MATLAB format (auto-saves to output/ with unique filename)
    matlab_output_path = session.export_to_matlab()
    if matlab_output_path:
        print(f"Saved MATLAB file: {matlab_output_path}")

    # Export results to CSV format (auto-saves to output/ with unique filename)
    csv_output_paths = session.export_to_csv()
    if csv_output_paths:
        print(f"Saved CSV file(s): {csv_output_paths}")

    # Save visualization to file (auto-saves to output/ with unique filename)
    saved_path = session.visualize_tracks()
    if saved_path:
        print(f"Saved plot image: {saved_path}")


def main_alternative() -> None:
    """Alternative approach using explicit exporters and visualizer.

    This demonstrates direct use of the exporter and visualizer classes
    for more control over the output process. All files are automatically
    saved to the 'output' directory with unique filenames.
    """
    from cam_track_gen import (
        AircraftTrackGenerator,
        CsvTrackResultExporter,
        MatlabTrackResultExporter,
        load_bayesian_network_model_from_file,
    )

    # Configuration
    simulation_duration_seconds = 250
    number_of_tracks = 10
    model_filename = "Light_Aircraft_Below_10000_ft_Data.mat"
    output_base_name = "Light_Aircraft_Below_10000_ft_Data"

    # Load the model
    model_data = load_bayesian_network_model_from_file(model_filename)
    if model_data is None:
        print(f"Failed to load model: {model_filename}")
        return

    # Create the track generator
    generator = AircraftTrackGenerator(model_data)

    # Generate tracks
    track_results, initial_conditions_df, _ = generator.generate_multiple_tracks(
        number_of_tracks=number_of_tracks,
        simulation_duration_seconds=simulation_duration_seconds,
        use_reproducible_seed=False,
    )

    if not track_results:
        print("No tracks were generated.")
        return

    print(f"Generated {len(track_results)} valid tracks")
    print(f"Initial conditions:\n{initial_conditions_df}")

    # Export to MATLAB (auto-saves to output/ with unique filename)
    matlab_exporter = MatlabTrackResultExporter()
    matlab_path = matlab_exporter.export_tracks(track_results, output_base_name)
    print(f"Exported to MATLAB: {matlab_path}")

    # Export to CSV (auto-saves to output/ with unique filename)
    csv_exporter = CsvTrackResultExporter()
    csv_paths = csv_exporter.export_tracks(track_results, output_base_name)
    print(f"Exported to CSV: {csv_paths}")

    # Save visualization to file (auto-saves to output/ with unique filename)
    plot_title = output_base_name.replace("_", " ")
    saved_path = TrackVisualizationRenderer.render_three_dimensional_tracks(
        track_results, plot_title, output_filename_base=output_base_name
    )
    if saved_path:
        print(f"Saved plot image: {saved_path}")


if __name__ == "__main__":
    main()
