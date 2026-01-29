"""Pytest-based performance tests for track generation.

These tests can be run with pytest to validate performance characteristics
and detect performance regressions.

Run with:
    uv run pytest tests/performance/test_performance.py -v
    uv run pytest tests/performance/test_performance.py -v --benchmark  # Full benchmark
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from cam_track_gen import (
    AircraftTrackGenerator,
    TrackGenerationSession,
    get_available_model_files,
    load_bayesian_network_model_from_file,
)

if TYPE_CHECKING:
    from cam_track_gen.data_classes import BayesianNetworkModelData


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def model_data() -> BayesianNetworkModelData | None:
    """Load model data once for all tests in module."""
    return load_bayesian_network_model_from_file("Light_Aircraft_Below_10000_ft_Data.mat")


@pytest.fixture(scope="module")
def track_generator(model_data: BayesianNetworkModelData | None) -> AircraftTrackGenerator | None:
    """Create a track generator for testing."""
    if model_data is None:
        return None
    return AircraftTrackGenerator(model_data)


@pytest.fixture(scope="module")
def session() -> TrackGenerationSession | None:
    """Create a session for testing."""
    return TrackGenerationSession.create_from_file("Light_Aircraft_Below_10000_ft_Data.mat")


# =============================================================================
# Performance Tests
# =============================================================================


class TestTrackGenerationPerformance:
    """Performance tests for track generation."""

    @pytest.mark.parametrize(
        ("number_of_tracks", "duration_seconds", "max_time_seconds"),
        [
            (1, 100, 5.0),  # Single short track
            (5, 250, 15.0),  # Multiple tracks
            (10, 250, 30.0),  # Standard benchmark configuration
        ],
    )
    def test_track_generation_time(
        self,
        session: TrackGenerationSession | None,
        number_of_tracks: int,
        duration_seconds: int,
        max_time_seconds: float,
    ) -> None:
        """Test that track generation completes within expected time."""
        if session is None:
            pytest.skip("Could not load model")

        start_time = time.perf_counter()

        tracks = session.generate_tracks(
            number_of_tracks=number_of_tracks,
            simulation_duration_seconds=duration_seconds,
            use_reproducible_seed=True,
        )

        elapsed_time = time.perf_counter() - start_time

        assert len(tracks) == number_of_tracks
        assert elapsed_time < max_time_seconds, f"Track generation took {elapsed_time:.2f}s, expected < {max_time_seconds}s"

        # Log performance info
        tracks_per_second = number_of_tracks / elapsed_time
        print(f"\nGenerated {number_of_tracks} tracks in {elapsed_time:.4f}s ({tracks_per_second:.2f} tracks/s)")

    def test_single_track_generation_performance(
        self,
        track_generator: AircraftTrackGenerator | None,
    ) -> None:
        """Test performance of generating a single track."""
        if track_generator is None:
            pytest.skip("Could not load model")

        # Time individual components
        timings: dict[str, float] = {}

        # Time initial condition sampling
        start = time.perf_counter()
        initial_conditions = track_generator.sampler.sample_initial_state_conditions()
        timings["initial_sampling"] = time.perf_counter() - start

        # Time transition sampling
        start = time.perf_counter()
        transitions = track_generator.sampler.sample_transition_state_sequence(initial_conditions, 250)
        timings["transition_sampling"] = time.perf_counter() - start

        # Time track conversion/simulation
        initial_conditions_dict = {label: int(initial_conditions[index]) for index, label in enumerate(track_generator.model_data.variable_labels)}

        start = time.perf_counter()
        track_result = track_generator._data_converter.convert_sampled_data_to_track(
            initial_conditions_dict,
            transitions,
            250,
        )
        timings["track_simulation"] = time.perf_counter() - start

        # Print timing breakdown
        total = sum(timings.values())
        print("\nSingle track generation timing breakdown:")
        print("-" * 50)
        for name, duration in timings.items():
            percentage = (duration / total) * 100
            print(f"  {name:.<30} {duration * 1000:>8.2f}ms ({percentage:>5.1f}%)")
        print("-" * 50)
        print(f"  {'Total':.<30} {total * 1000:>8.2f}ms")

        # Ensure track was generated
        assert track_result is not None
        assert "time" in track_result
        assert len(track_result["time"]) > 0

    def test_throughput_benchmark(
        self,
        session: TrackGenerationSession | None,
    ) -> None:
        """Benchmark track generation throughput."""
        if session is None:
            pytest.skip("Could not load model")

        # Warmup
        session.generate_tracks(
            number_of_tracks=2,
            simulation_duration_seconds=50,
            use_reproducible_seed=True,
        )

        # Benchmark
        iterations = 3
        tracks_per_iteration = 5
        duration = 250

        times: list[float] = []
        for i in range(iterations):
            start = time.perf_counter()
            tracks = session.generate_tracks(
                number_of_tracks=tracks_per_iteration,
                simulation_duration_seconds=duration,
                use_reproducible_seed=True,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            assert len(tracks) == tracks_per_iteration

        import statistics

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0

        throughput = tracks_per_iteration / mean_time

        print("\nThroughput benchmark results:")
        print(f"  Mean time: {mean_time:.4f}s ± {std_time:.4f}s")
        print(f"  Throughput: {throughput:.2f} tracks/second")
        print(f"  Time per track: {mean_time / tracks_per_iteration * 1000:.2f}ms")


class TestComponentPerformance:
    """Performance tests for individual components."""

    def test_bayesian_network_sampling_performance(
        self,
        track_generator: AircraftTrackGenerator | None,
    ) -> None:
        """Test Bayesian network sampling performance."""
        if track_generator is None:
            pytest.skip("Could not load model")

        sampler = track_generator.sampler
        iterations = 100

        # Benchmark initial condition sampling
        start = time.perf_counter()
        for _ in range(iterations):
            sampler.sample_initial_state_conditions()
        initial_time = time.perf_counter() - start

        print("\nInitial condition sampling:")
        print(f"  {iterations} samples in {initial_time:.4f}s")
        print(f"  {initial_time / iterations * 1000:.4f}ms per sample")

        # Benchmark transition sampling
        initial_conditions = sampler.sample_initial_state_conditions()
        sequence_length = 250

        start = time.perf_counter()
        for _ in range(iterations):
            sampler.sample_transition_state_sequence(initial_conditions, sequence_length)
        transition_time = time.perf_counter() - start

        print(f"\nTransition sampling ({sequence_length} steps):")
        print(f"  {iterations} sequences in {transition_time:.4f}s")
        print(f"  {transition_time / iterations * 1000:.4f}ms per sequence")

    def test_dynamics_integration_performance(
        self,
        track_generator: AircraftTrackGenerator | None,
    ) -> None:
        """Test dynamics integration performance."""
        if track_generator is None:
            pytest.skip("Could not load model")

        from cam_track_gen.data_classes import AircraftKinematicState, SimulationControlParameters
        from cam_track_gen.dynamics import AircraftDynamicsIntegrator

        # Create test state and parameters
        config = track_generator.configuration
        params = SimulationControlParameters.from_performance_limits(config.performance_limits)
        integrator = AircraftDynamicsIntegrator(params)

        initial_state = AircraftKinematicState(
            velocity_feet_per_second=200.0,
            north_position_feet=0.0,
            east_position_feet=0.0,
            altitude_feet=5000.0,
            heading_angle_radians=0.0,
            pitch_angle_radians=0.0,
            bank_angle_radians=0.0,
            acceleration_feet_per_second_squared=0.0,
        )

        # Benchmark integration steps
        iterations = 10000
        state = initial_state

        start = time.perf_counter()
        for _ in range(iterations):
            state = integrator.integrate_single_time_step(
                state,
                acceleration_command=0.5,
                vertical_rate_command=1.0,
                heading_change_command=0.01,
            )
        elapsed = time.perf_counter() - start

        print("\nDynamics integration:")
        print(f"  {iterations} integration steps in {elapsed:.4f}s")
        print(f"  {elapsed / iterations * 1000000:.2f}µs per step")
        print(f"  {iterations / elapsed:.0f} steps/second")


class TestMemoryPerformance:
    """Memory usage performance tests."""

    def test_track_generation_memory(
        self,
        session: TrackGenerationSession | None,
    ) -> None:
        """Test memory usage during track generation."""
        if session is None:
            pytest.skip("Could not load model")

        import sys

        # Generate tracks and measure memory
        tracks = session.generate_tracks(
            number_of_tracks=10,
            simulation_duration_seconds=250,
            use_reproducible_seed=True,
        )

        # Calculate approximate memory usage
        total_size = 0
        for track in tracks:
            for key, value in track.items():
                total_size += sys.getsizeof(value)
                if hasattr(value, "nbytes"):
                    total_size += value.nbytes

        print("\nMemory usage for 10 tracks (250s each):")
        print(f"  Approximate size: {total_size / 1024:.2f} KB")
        print(f"  Per track: {total_size / 10 / 1024:.2f} KB")


# =============================================================================
# Regression Tests
# =============================================================================


class TestPerformanceRegression:
    """Tests to detect performance regressions."""

    # These thresholds should be updated based on baseline measurements
    SINGLE_TRACK_MAX_TIME_SECONDS = 3.0
    TEN_TRACKS_MAX_TIME_SECONDS = 20.0

    def test_single_track_regression(
        self,
        session: TrackGenerationSession | None,
    ) -> None:
        """Ensure single track generation doesn't regress."""
        if session is None:
            pytest.skip("Could not load model")

        start = time.perf_counter()
        tracks = session.generate_tracks(
            number_of_tracks=1,
            simulation_duration_seconds=250,
            use_reproducible_seed=True,
        )
        elapsed = time.perf_counter() - start

        assert len(tracks) == 1
        assert elapsed < self.SINGLE_TRACK_MAX_TIME_SECONDS, (
            f"Single track generation took {elapsed:.2f}s, threshold is {self.SINGLE_TRACK_MAX_TIME_SECONDS}s"
        )

    def test_batch_generation_regression(
        self,
        session: TrackGenerationSession | None,
    ) -> None:
        """Ensure batch track generation doesn't regress."""
        if session is None:
            pytest.skip("Could not load model")

        start = time.perf_counter()
        tracks = session.generate_tracks(
            number_of_tracks=10,
            simulation_duration_seconds=250,
            use_reproducible_seed=True,
        )
        elapsed = time.perf_counter() - start

        assert len(tracks) == 10
        assert elapsed < self.TEN_TRACKS_MAX_TIME_SECONDS, (
            f"10 track generation took {elapsed:.2f}s, threshold is {self.TEN_TRACKS_MAX_TIME_SECONDS}s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
