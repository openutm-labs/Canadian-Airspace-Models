"""Benchmarking utilities for track generation performance measurement.

This module provides tools for running reproducible benchmarks
and comparing performance across different versions or optimizations.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    model_filename: str = "Light_Aircraft_Below_10000_ft_Data.mat"
    number_of_tracks: int = 10
    simulation_duration_seconds: int = 250
    warmup_iterations: int = 1
    benchmark_iterations: int = 5
    use_reproducible_seed: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    timestamp: str
    total_times_seconds: list[float]
    tracks_generated: int
    tracks_per_second: float
    mean_time_seconds: float
    std_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    median_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_timings(
        cls,
        config: BenchmarkConfig,
        timings: list[float],
        tracks_per_iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Create a BenchmarkResult from raw timing data."""
        total_tracks = len(timings) * tracks_per_iteration
        mean_time = statistics.mean(timings)

        return cls(
            config=config,
            timestamp=datetime.now().isoformat(),
            total_times_seconds=timings,
            tracks_generated=total_tracks,
            tracks_per_second=tracks_per_iteration / mean_time,
            mean_time_seconds=mean_time,
            std_time_seconds=statistics.stdev(timings) if len(timings) > 1 else 0.0,
            min_time_seconds=min(timings),
            max_time_seconds=max(timings),
            median_time_seconds=statistics.median(timings),
            metadata=metadata or {},
        )

    def format_report(self) -> str:
        """Format a human-readable benchmark report."""
        lines = [
            "=" * 80,
            "BENCHMARK RESULTS",
            "=" * 80,
            f"Timestamp: {self.timestamp}",
            "",
            "Configuration:",
            f"  Model: {self.config.model_filename}",
            f"  Tracks per iteration: {self.config.number_of_tracks}",
            f"  Simulation duration: {self.config.simulation_duration_seconds}s",
            f"  Benchmark iterations: {self.config.benchmark_iterations}",
            "",
            "Results:",
            f"  Total tracks generated: {self.tracks_generated}",
            f"  Throughput: {self.tracks_per_second:.4f} tracks/second",
            "",
            "Timing Statistics:",
            f"  Mean time: {self.mean_time_seconds:.4f}s",
            f"  Std deviation: {self.std_time_seconds:.4f}s",
            f"  Min time: {self.min_time_seconds:.4f}s",
            f"  Max time: {self.max_time_seconds:.4f}s",
            f"  Median time: {self.median_time_seconds:.4f}s",
            "",
            "Per-Iteration Times:",
        ]

        for i, t in enumerate(self.total_times_seconds, 1):
            lines.append(f"  Iteration {i}: {t:.4f}s")

        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return {
            "config": {
                "model_filename": self.config.model_filename,
                "number_of_tracks": self.config.number_of_tracks,
                "simulation_duration_seconds": self.config.simulation_duration_seconds,
                "warmup_iterations": self.config.warmup_iterations,
                "benchmark_iterations": self.config.benchmark_iterations,
                "use_reproducible_seed": self.config.use_reproducible_seed,
            },
            "timestamp": self.timestamp,
            "total_times_seconds": self.total_times_seconds,
            "tracks_generated": self.tracks_generated,
            "tracks_per_second": self.tracks_per_second,
            "mean_time_seconds": self.mean_time_seconds,
            "std_time_seconds": self.std_time_seconds,
            "min_time_seconds": self.min_time_seconds,
            "max_time_seconds": self.max_time_seconds,
            "median_time_seconds": self.median_time_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create from a dictionary (for JSON deserialization)."""
        config_data = data["config"]
        config = BenchmarkConfig(
            model_filename=config_data["model_filename"],
            number_of_tracks=config_data["number_of_tracks"],
            simulation_duration_seconds=config_data["simulation_duration_seconds"],
            warmup_iterations=config_data["warmup_iterations"],
            benchmark_iterations=config_data["benchmark_iterations"],
            use_reproducible_seed=config_data["use_reproducible_seed"],
        )
        return cls(
            config=config,
            timestamp=data["timestamp"],
            total_times_seconds=data["total_times_seconds"],
            tracks_generated=data["tracks_generated"],
            tracks_per_second=data["tracks_per_second"],
            mean_time_seconds=data["mean_time_seconds"],
            std_time_seconds=data["std_time_seconds"],
            min_time_seconds=data["min_time_seconds"],
            max_time_seconds=data["max_time_seconds"],
            median_time_seconds=data["median_time_seconds"],
            metadata=data.get("metadata", {}),
        )


class TrackGenerationBenchmark:
    """Benchmark runner for track generation."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration (uses defaults if not provided).
        """
        self.config = config or BenchmarkConfig()
        self._session: Any = None

    def setup(self) -> bool:
        """Set up the benchmark by loading the model.

        Returns:
            True if setup was successful, False otherwise.
        """
        from cam_track_gen import TrackGenerationSession

        self._session = TrackGenerationSession.create_from_file(self.config.model_filename)
        return self._session is not None

    def run_single_iteration(self) -> float:
        """Run a single benchmark iteration.

        Returns:
            Time taken in seconds.
        """
        if self._session is None:
            raise RuntimeError("Benchmark not set up. Call setup() first.")

        start_time = time.perf_counter()

        self._session.generate_tracks(
            number_of_tracks=self.config.number_of_tracks,
            simulation_duration_seconds=self.config.simulation_duration_seconds,
            use_reproducible_seed=self.config.use_reproducible_seed,
        )

        return time.perf_counter() - start_time

    def run(self, metadata: dict[str, Any] | None = None) -> BenchmarkResult:
        """Run the complete benchmark.

        Args:
            metadata: Optional metadata to include in results.

        Returns:
            BenchmarkResult with timing statistics.
        """
        if not self.setup():
            raise RuntimeError(f"Failed to load model: {self.config.model_filename}")

        # Warmup iterations
        print(f"Running {self.config.warmup_iterations} warmup iteration(s)...")
        for i in range(self.config.warmup_iterations):
            warmup_time = self.run_single_iteration()
            print(f"  Warmup {i + 1}: {warmup_time:.4f}s")

        # Benchmark iterations
        print(f"\nRunning {self.config.benchmark_iterations} benchmark iteration(s)...")
        timings: list[float] = []
        for i in range(self.config.benchmark_iterations):
            iteration_time = self.run_single_iteration()
            timings.append(iteration_time)
            print(f"  Iteration {i + 1}: {iteration_time:.4f}s")

        result = BenchmarkResult.from_timings(
            config=self.config,
            timings=timings,
            tracks_per_iteration=self.config.number_of_tracks,
            metadata=metadata,
        )

        return result


@dataclass
class BenchmarkComparison:
    """Comparison between two benchmark results."""

    baseline: BenchmarkResult
    optimized: BenchmarkResult
    speedup_factor: float
    time_saved_seconds: float
    time_saved_percentage: float

    @classmethod
    def compare(cls, baseline: BenchmarkResult, optimized: BenchmarkResult) -> BenchmarkComparison:
        """Compare two benchmark results.

        Args:
            baseline: Baseline benchmark result.
            optimized: Optimized benchmark result.

        Returns:
            BenchmarkComparison with speedup metrics.
        """
        speedup = baseline.mean_time_seconds / optimized.mean_time_seconds
        time_saved = baseline.mean_time_seconds - optimized.mean_time_seconds
        percentage = (time_saved / baseline.mean_time_seconds) * 100

        return cls(
            baseline=baseline,
            optimized=optimized,
            speedup_factor=speedup,
            time_saved_seconds=time_saved,
            time_saved_percentage=percentage,
        )

    def format_report(self) -> str:
        """Format a comparison report."""
        lines = [
            "=" * 80,
            "BENCHMARK COMPARISON",
            "=" * 80,
            "",
            "Baseline:",
            f"  Mean time: {self.baseline.mean_time_seconds:.4f}s",
            f"  Throughput: {self.baseline.tracks_per_second:.4f} tracks/second",
            "",
            "Optimized:",
            f"  Mean time: {self.optimized.mean_time_seconds:.4f}s",
            f"  Throughput: {self.optimized.tracks_per_second:.4f} tracks/second",
            "",
            "Improvement:",
            f"  Speedup factor: {self.speedup_factor:.2f}x",
            f"  Time saved: {self.time_saved_seconds:.4f}s ({self.time_saved_percentage:.1f}%)",
            f"  Throughput increase: {self.optimized.tracks_per_second - self.baseline.tracks_per_second:.4f} tracks/second",
            "=" * 80,
        ]
        return "\n".join(lines)


def save_benchmark_result(result: BenchmarkResult, output_path: Path) -> None:
    """Save benchmark result to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_benchmark_result(input_path: Path) -> BenchmarkResult:
    """Load benchmark result from a JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    return BenchmarkResult.from_dict(data)


def run_benchmark(
    model_filename: str = "Light_Aircraft_Below_10000_ft_Data.mat",
    number_of_tracks: int = 10,
    simulation_duration_seconds: int = 250,
    benchmark_iterations: int = 5,
    warmup_iterations: int = 1,
) -> BenchmarkResult:
    """Convenience function to run a benchmark with custom parameters.

    Args:
        model_filename: Name of the model file.
        number_of_tracks: Tracks to generate per iteration.
        simulation_duration_seconds: Duration of each track.
        benchmark_iterations: Number of timed iterations.
        warmup_iterations: Number of warmup iterations.

    Returns:
        BenchmarkResult with timing statistics.
    """
    config = BenchmarkConfig(
        model_filename=model_filename,
        number_of_tracks=number_of_tracks,
        simulation_duration_seconds=simulation_duration_seconds,
        benchmark_iterations=benchmark_iterations,
        warmup_iterations=warmup_iterations,
    )

    benchmark = TrackGenerationBenchmark(config)
    return benchmark.run()
