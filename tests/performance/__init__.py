"""Performance testing framework for Canadian Airspace Models track generation.

This package provides utilities for profiling, benchmarking, and measuring
performance improvements in the track generation workflow.
"""

# Note: Imports are done lazily or via relative imports in the individual modules
# to avoid circular import issues when running as a package vs standalone scripts.

__all__ = [
    "ProfileResult",
    "TrackGenerationProfiler",
    "BenchmarkConfig",
    "BenchmarkResult",
    "TrackGenerationBenchmark",
]


def __getattr__(name: str):  # noqa: ANN202
    """Lazy loading of performance testing classes."""
    if name in ("ProfileResult", "TrackGenerationProfiler"):
        from tests.performance.profiler import ProfileResult, TrackGenerationProfiler

        if name == "ProfileResult":
            return ProfileResult
        return TrackGenerationProfiler

    if name in ("BenchmarkConfig", "BenchmarkResult", "TrackGenerationBenchmark"):
        from tests.performance.benchmark import (
            BenchmarkConfig,
            BenchmarkResult,
            TrackGenerationBenchmark,
        )

        if name == "BenchmarkConfig":
            return BenchmarkConfig
        if name == "BenchmarkResult":
            return BenchmarkResult
        return TrackGenerationBenchmark

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
