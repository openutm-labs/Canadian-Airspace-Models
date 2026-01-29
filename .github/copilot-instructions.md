# Canadian Airspace Models - Copilot Instructions

## Project Overview

This repository provides Python tools for generating simulated aircraft tracks using Bayesian network models trained on Canadian airspace data. The core library (`cam_track_gen`) samples from MATLAB-based statistical models to produce realistic flight trajectories for RPAS (drone) integration safety research.

## Architecture

```
src/cam_track_gen/
├── track_generation_tool.py  # Core module (~2200 lines, single-file architecture)
├── data/*.mat                # Bundled Bayesian network models (12 aircraft types)
└── __init__.py               # Public API exports
```

**Key components in `track_generation_tool.py`:**
- `TrackGenerationSession` - High-level session-based API (preferred)
- `AircraftTrackGenerator` - Orchestrates track generation pipeline
- `BayesianNetworkStateSampler` - Samples from DAG-based probability distributions
- `AircraftDynamicsIntegrator` - Physics simulation with backwards Euler integration
- `TrackResultData` (TypedDict) - Standard track output format with units in feet/radians

**Data flow:** Model file (.mat) → `BayesianNetworkModelData` → Initial state sampling → Transition sampling → Dynamics integration → `TrackResultData`

## Development Commands

```bash
# Run track generation demo
uv run generate_tracks.py

# Run tests (pytest with fixtures in tests/conftest.py)
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=cam_track_gen --cov-report=term-missing
```

## Code Conventions

### Units & Naming
- All spatial values in **feet**, velocities in **feet/second**, angles in **radians**
- Variable names explicitly include units: `altitude_feet`, `speed_feet_per_second`, `heading_angle_radians`
- Use descriptive multi-word names following module patterns (e.g., `maximum_pitch_rate_radians_per_second`)

### Type Hints
- Full type annotations required (Python 3.12+)
- Use `NDArray[np.floating[Any]]` for numpy arrays
- Use `TypedDict` for structured dictionaries (see `TrackResultData`)
- Use `Protocol` for interface definitions (see `ProbabilityDistributionSampler`)

### Constants Organization
Follow the module's pattern of grouping constants into classes:
```python
class UnitConversionConstants:
    KNOTS_TO_FEET_PER_SECOND: Final[float] = 1.68780972222222
```

### Code Style
- Line length: 150 characters (`pyproject.toml` → `tool.ruff`)
- Docstrings for all public functions with Args/Returns sections
- Import sorting via ruff (`fixable = ["I"]`)

## Model Files

Bundled `.mat` files in `src/cam_track_gen/data/` contain:
- `DAG_Initial`, `DAG_Transition` - Bayesian network adjacency matrices
- `N_initial`, `N_transition` - Frequency tables for probability distributions
- `Cut_Points` - Discretization boundaries for continuous variables
- `resample_rate` - Temporal resampling rates

Use `get_available_model_files()` to list available models by name.

## Testing Patterns

Tests use pytest with parametrize for thorough coverage. Key fixtures in `tests/conftest.py`:
- `package_data_directory` - Path to bundled model files
- Component fixtures like `sample_velocity_limits`, `sample_performance_limits`

Mock external dependencies with `unittest.mock.patch` when testing components in isolation.

## Output Handling

All exports auto-save to `output/` directory with collision-safe naming (appends `_1`, `_2`, etc.):
```python
session.export_to_matlab()  # → output/Model_Name_Result.mat
session.export_to_csv()     # → output/Model_Name_Result.csv (splits at 1M rows)
session.visualize_tracks()  # → output/Model_Name.png
```
