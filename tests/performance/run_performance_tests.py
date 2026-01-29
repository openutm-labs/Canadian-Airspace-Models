#!/usr/bin/env python3
"""Run performance profiling and benchmarking for track generation.

This script provides a command-line interface for running performance
tests and generating reports.

Usage:
    # Run full benchmark
    uv run python tests/performance/run_performance_tests.py benchmark

    # Run profiler to identify bottlenecks
    uv run python tests/performance/run_performance_tests.py profile

    # Run quick test (fewer iterations)
    uv run python tests/performance/run_performance_tests.py quick

    # Compare baseline vs optimized results
    uv run python tests/performance/run_performance_tests.py compare baseline.json optimized.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.performance.benchmark import (  # noqa: E402
    BenchmarkComparison,
    BenchmarkConfig,
    BenchmarkResult,
    TrackGenerationBenchmark,
    load_benchmark_result,
    save_benchmark_result,
)
from tests.performance.profiler import (  # noqa: E402
    ProfileResult,
    profile_track_generation,
    save_profile_results,
)


def get_output_directory() -> Path:
    """Get the performance results output directory."""
    output_dir = PROJECT_ROOT / "tests" / "performance" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_benchmark(
    number_of_tracks: int = 10,
    simulation_duration: int = 250,
    iterations: int = 5,
    warmup: int = 1,
    save: bool = True,
    label: str = "",
) -> BenchmarkResult:
    """Run a benchmark and optionally save results.

    Args:
        number_of_tracks: Number of tracks per iteration.
        simulation_duration: Duration of each track in seconds.
        iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        save: Whether to save results to file.
        label: Optional label for the results file.

    Returns:
        BenchmarkResult with timing data.
    """
    print("\n" + "=" * 80)
    print("TRACK GENERATION BENCHMARK")
    print("=" * 80)
    print("Configuration:")
    print(f"  Tracks per iteration: {number_of_tracks}")
    print(f"  Simulation duration: {simulation_duration}s")
    print(f"  Benchmark iterations: {iterations}")
    print(f"  Warmup iterations: {warmup}")
    print("")

    config = BenchmarkConfig(
        number_of_tracks=number_of_tracks,
        simulation_duration_seconds=simulation_duration,
        benchmark_iterations=iterations,
        warmup_iterations=warmup,
    )

    benchmark = TrackGenerationBenchmark(config)

    import platform
    import sys

    metadata = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "label": label,
    }

    result = benchmark.run(metadata=metadata)

    print("\n" + result.format_report())

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{label}" if label else ""
        filename = f"benchmark_{timestamp}{label_suffix}.json"
        output_path = get_output_directory() / filename
        save_benchmark_result(result, output_path)
        print(f"\nResults saved to: {output_path}")

    return result


def run_profiler(
    number_of_tracks: int = 10,
    simulation_duration: int = 250,
    save: bool = True,
    label: str = "",
) -> ProfileResult:
    """Run the profiler to identify performance bottlenecks.

    Args:
        number_of_tracks: Number of tracks to generate.
        simulation_duration: Duration of each track in seconds.
        save: Whether to save results to file.
        label: Optional label for the results file.

    Returns:
        ProfileResult with function-level timing data.
    """
    print("\n" + "=" * 80)
    print("TRACK GENERATION PROFILER")
    print("=" * 80)
    print("Configuration:")
    print(f"  Number of tracks: {number_of_tracks}")
    print(f"  Simulation duration: {simulation_duration}s")
    print("")

    print("Running profiler...")
    result = profile_track_generation(
        model_filename="Light_Aircraft_Below_10000_ft_Data.mat",
        number_of_tracks=number_of_tracks,
        simulation_duration_seconds=simulation_duration,
    )

    print("\n" + result.format_report())

    # Identify key functions from our codebase
    print("\n" + "=" * 80)
    print("KEY FUNCTIONS FROM CAM_TRACK_GEN PACKAGE")
    print("=" * 80)

    cam_functions = result.get_functions_by_name("cam_track_gen")
    cam_functions = sorted(cam_functions, key=lambda x: x.cumulative_time_seconds, reverse=True)

    print(f"\n{'Function':<60} {'Calls':>8} {'Cum(s)':>10} {'Per Call(ms)':>12}")
    print("-" * 92)

    for func in cam_functions[:30]:
        name = func.function_name
        if len(name) > 58:
            name = "..." + name[-55:]
        print(f"{name:<60} {func.call_count:>8} {func.cumulative_time_seconds:>10.4f} {func.time_per_call_seconds * 1000:>12.4f}")

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{label}" if label else ""
        filename = f"profile_{timestamp}{label_suffix}.txt"
        output_path = get_output_directory() / filename
        save_profile_results(result, output_path)
        print(f"\nProfile results saved to: {output_path}")

    return result


def run_quick_test() -> BenchmarkResult:
    """Run a quick benchmark with minimal iterations."""
    print("\n" + "=" * 80)
    print("QUICK PERFORMANCE TEST")
    print("=" * 80)
    return run_benchmark(
        number_of_tracks=5,
        simulation_duration=100,
        iterations=3,
        warmup=1,
        save=False,
        label="quick",
    )


def compare_results(baseline_path: Path, optimized_path: Path) -> BenchmarkComparison:
    """Compare two benchmark results.

    Args:
        baseline_path: Path to baseline results JSON.
        optimized_path: Path to optimized results JSON.

    Returns:
        BenchmarkComparison with speedup metrics.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    baseline = load_benchmark_result(baseline_path)
    optimized = load_benchmark_result(optimized_path)

    comparison = BenchmarkComparison.compare(baseline, optimized)
    print("\n" + comparison.format_report())

    return comparison


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Track Generation Performance Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("-n", "--tracks", type=int, default=10, help="Number of tracks per iteration")
    bench_parser.add_argument("-d", "--duration", type=int, default=250, help="Simulation duration (seconds)")
    bench_parser.add_argument("-i", "--iterations", type=int, default=5, help="Benchmark iterations")
    bench_parser.add_argument("-w", "--warmup", type=int, default=1, help="Warmup iterations")
    bench_parser.add_argument("-l", "--label", type=str, default="", help="Label for results file")
    bench_parser.add_argument("--no-save", action="store_true", help="Don't save results")

    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Run profiler")
    profile_parser.add_argument("-n", "--tracks", type=int, default=10, help="Number of tracks")
    profile_parser.add_argument("-d", "--duration", type=int, default=250, help="Simulation duration (seconds)")
    profile_parser.add_argument("-l", "--label", type=str, default="", help="Label for results file")
    profile_parser.add_argument("--no-save", action="store_true", help="Don't save results")

    # Quick test command
    subparsers.add_parser("quick", help="Run quick test")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument("baseline", type=Path, help="Path to baseline results JSON")
    compare_parser.add_argument("optimized", type=Path, help="Path to optimized results JSON")

    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark(
            number_of_tracks=args.tracks,
            simulation_duration=args.duration,
            iterations=args.iterations,
            warmup=args.warmup,
            save=not args.no_save,
            label=args.label,
        )
    elif args.command == "profile":
        run_profiler(
            number_of_tracks=args.tracks,
            simulation_duration=args.duration,
            save=not args.no_save,
            label=args.label,
        )
    elif args.command == "quick":
        run_quick_test()
    elif args.command == "compare":
        compare_results(args.baseline, args.optimized)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
