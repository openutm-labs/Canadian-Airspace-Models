# Canadian Airspace Models - Copilot Instructions

## Project Overview

Python library for generating simulated aircraft tracks using Bayesian network models trained on Canadian airspace data. Used for RPAS (drone) integration safety research.

## Architecture

```
src/cam_track_gen/
├── generator.py        # TrackGenerationSession, AircraftTrackGenerator (main entry points)
├── bayesian_network.py # BayesianNetworkStateSampler, probability sampling
├── dynamics.py         # AircraftDynamicsIntegrator, physics simulation
├── types.py            # TrackResultData (TypedDict), Protocol definitions
├── constants.py        # Grouped constants (UnitConversionConstants, PhysicsConstants)
├── data_classes.py     # Frozen dataclasses (VelocityLimits, AltitudeBoundary, etc.)
├── exporters.py        # CSV/MATLAB export (CsvTrackResultExporter, MatlabTrackResultExporter)
├── validation.py       # ConstraintBasedTrackValidator
├── visualization.py    # TrackVisualizationRenderer
└── data/*.mat          # Bundled Bayesian network models (12 aircraft types)
```

**Data flow:** `.mat` file → `BayesianNetworkModelData` → Initial state sampling → Transition sampling → Dynamics integration → `TrackResultData`

**Preferred API:** Use `TrackGenerationSession` for high-level operations. See [generate_tracks.py](../generate_tracks.py) for usage.

## Development Commands

```bash
uv run generate_tracks.py                                      # Demo script
uv run pytest tests/                                           # Run tests
uv run pytest tests/ --cov=cam_track_gen --cov-report=term-missing  # With coverage
ruff format && ruff check --fix                                # Format and lint
```

## Code Conventions

### Units & Naming (Critical)
- **Spatial:** feet | **Velocity:** feet/second | **Angles:** radians
- Variable names MUST include units: `altitude_feet`, `speed_feet_per_second`, `heading_angle_radians`
- Multi-word descriptive names: `maximum_pitch_rate_radians_per_second`

### Type Hints (Python 3.12+)
```python
from numpy.typing import NDArray
def process(data: NDArray[np.floating[Any]]) -> TrackResultData: ...
```
- Use `TypedDict` for structured dicts, `Protocol` for interfaces, frozen `@dataclass` for value objects

### Constants Pattern
```python
class UnitConversionConstants:
    KNOTS_TO_FEET_PER_SECOND: Final[float] = 1.68780972222222
```

### Style
- Line length: 150 chars | Docstrings with Args/Returns | ruff for imports (`fixable = ["I"]`)

## Testing Patterns

Tests in [tests/test_track_generation_tool.py](../tests/test_track_generation_tool.py) use pytest parametrize extensively.

```python
@pytest.fixture
def sample_velocity_limits() -> VelocityLimits:
    return VelocityLimits(minimum_feet_per_second=50.0, maximum_feet_per_second=500.0)
```

Key fixtures from `conftest.py`: `package_data_directory`, `test_data_directory`

## Output Handling

Auto-saves to `output/` with collision-safe naming (`_1`, `_2`, etc.):
```python
session.export_to_matlab()  # → output/Model_Name_Result.mat
session.export_to_csv()     # → output/Model_Name_Result.csv (splits at 1M rows)
session.visualize_tracks()  # → output/Model_Name.png
```

## Model Files

Use `get_available_model_files()` to list bundled models. `.mat` files contain:
- `DAG_Initial`, `DAG_Transition` - Bayesian network adjacency matrices
- `N_initial`, `N_transition` - Frequency tables
- `Cut_Points` - Discretization boundaries (note: some files have legacy typo `Aceleration`)
