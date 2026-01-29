"""Profiling utilities for track generation performance analysis.

This module provides tools for profiling the track generation workflow
to identify performance bottlenecks and measure function-level timings.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class FunctionTiming:
    """Timing information for a single function."""

    function_name: str
    total_time_seconds: float
    cumulative_time_seconds: float
    call_count: int
    time_per_call_seconds: float


@dataclass
class ProfileResult:
    """Results from a profiling session."""

    total_time_seconds: float
    function_timings: list[FunctionTiming]
    raw_stats: pstats.Stats | None = None
    profile_output: str = ""

    def get_top_functions(self, count: int = 20) -> list[FunctionTiming]:
        """Get the top N functions by cumulative time."""
        return sorted(self.function_timings, key=lambda x: x.cumulative_time_seconds, reverse=True)[:count]

    def get_functions_by_name(self, name_pattern: str) -> list[FunctionTiming]:
        """Get functions matching a name pattern."""
        return [f for f in self.function_timings if name_pattern in f.function_name]

    def format_report(self, top_count: int = 20) -> str:
        """Format a human-readable report of profiling results."""
        lines = [
            "=" * 80,
            "PROFILING RESULTS",
            "=" * 80,
            f"Total execution time: {self.total_time_seconds:.4f} seconds",
            "",
            f"Top {top_count} functions by cumulative time:",
            "-" * 80,
            f"{'Function':<50} {'Calls':>8} {'Total(s)':>10} {'Cum(s)':>10} {'Per Call':>10}",
            "-" * 80,
        ]

        for func in self.get_top_functions(top_count):
            name = func.function_name[:48] + ".." if len(func.function_name) > 50 else func.function_name
            lines.append(
                f"{name:<50} {func.call_count:>8} {func.total_time_seconds:>10.4f} "
                f"{func.cumulative_time_seconds:>10.4f} {func.time_per_call_seconds:>10.6f}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


class TrackGenerationProfiler:
    """Profiler for track generation workflow."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.profiler: cProfile.Profile | None = None
        self._start_time: float = 0
        self._end_time: float = 0

    def start(self) -> None:
        """Start profiling."""
        self.profiler = cProfile.Profile()
        self._start_time = time.perf_counter()
        self.profiler.enable()

    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        if self.profiler is None:
            raise RuntimeError("Profiler was not started")

        self.profiler.disable()
        self._end_time = time.perf_counter()

        # Capture stats
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats()

        # Parse function timings
        function_timings = self._parse_stats(stats)

        return ProfileResult(
            total_time_seconds=self._end_time - self._start_time,
            function_timings=function_timings,
            raw_stats=stats,
            profile_output=stream.getvalue(),
        )

    def _parse_stats(self, stats: pstats.Stats) -> list[FunctionTiming]:
        """Parse pstats.Stats into FunctionTiming objects."""
        timings: list[FunctionTiming] = []

        for (filename, line, func_name), (cc, nc, tt, ct, _) in stats.stats.items():
            # Format function name
            if filename:
                short_filename = Path(filename).name
                full_name = f"{short_filename}:{line}({func_name})"
            else:
                full_name = func_name

            time_per_call = ct / nc if nc > 0 else 0

            timings.append(
                FunctionTiming(
                    function_name=full_name,
                    total_time_seconds=tt,
                    cumulative_time_seconds=ct,
                    call_count=nc,
                    time_per_call_seconds=time_per_call,
                )
            )

        return timings

    def profile_function(self, func: F) -> F:
        """Decorator to profile a function."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.start()
            try:
                result = func(*args, **kwargs)
            finally:
                self.stop()
            return result

        return wrapper  # type: ignore[return-value]


@dataclass
class TimingContext:
    """Context manager for timing code blocks."""

    name: str
    elapsed_seconds: float = 0.0
    _start_time: float = field(default=0.0, repr=False)

    def __enter__(self) -> TimingContext:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start_time


class FunctionTimer:
    """Utility class to time individual functions and accumulate results."""

    def __init__(self) -> None:
        """Initialize the timer."""
        self.timings: dict[str, list[float]] = {}

    def time(self, name: str) -> TimingContext:
        """Create a timing context for a named operation."""
        context = TimingContext(name)
        if name not in self.timings:
            self.timings[name] = []
        return context

    def record(self, context: TimingContext) -> None:
        """Record a timing from a context."""
        self.timings[context.name].append(context.elapsed_seconds)

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all timed operations."""
        summary: dict[str, dict[str, float]] = {}

        for name, times in self.timings.items():
            if times:
                times_array = np.array(times)
                summary[name] = {
                    "count": len(times),
                    "total_seconds": float(np.sum(times_array)),
                    "mean_seconds": float(np.mean(times_array)),
                    "std_seconds": float(np.std(times_array)),
                    "min_seconds": float(np.min(times_array)),
                    "max_seconds": float(np.max(times_array)),
                    "median_seconds": float(np.median(times_array)),
                }

        return summary

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        summary = self.get_summary()
        lines = [
            "=" * 80,
            "TIMING SUMMARY",
            "=" * 80,
            f"{'Operation':<40} {'Count':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'Std(ms)':>10}",
            "-" * 80,
        ]

        for name, stats in sorted(summary.items(), key=lambda x: x[1]["total_seconds"], reverse=True):
            lines.append(
                f"{name:<40} {stats['count']:>8} {stats['total_seconds']:>10.4f} "
                f"{stats['mean_seconds'] * 1000:>10.4f} {stats['std_seconds'] * 1000:>10.4f}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)


def profile_track_generation(
    model_filename: str,
    number_of_tracks: int = 10,
    simulation_duration_seconds: int = 250,
) -> ProfileResult:
    """Profile the entire track generation workflow.

    Args:
        model_filename: Name of the model file to use.
        number_of_tracks: Number of tracks to generate.
        simulation_duration_seconds: Duration of each track.

    Returns:
        ProfileResult with detailed timing information.
    """
    from cam_track_gen import TrackGenerationSession

    profiler = TrackGenerationProfiler()

    # Create session outside of profiling to focus on generation
    session = TrackGenerationSession.create_from_file(model_filename)
    if session is None:
        raise ValueError(f"Failed to load model: {model_filename}")

    profiler.start()

    try:
        session.generate_tracks(
            number_of_tracks=number_of_tracks,
            simulation_duration_seconds=simulation_duration_seconds,
            use_reproducible_seed=True,
        )
    finally:
        result = profiler.stop()

    return result


def save_profile_results(result: ProfileResult, output_path: Path) -> None:
    """Save profiling results to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(result.format_report(top_count=50))
        f.write("\n\n")
        f.write("=" * 80)
        f.write("\nRAW PROFILE OUTPUT\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(result.profile_output)
