# Performance Baseline Results

## Date: 2026-01-29

## Test Configuration
- Model: `Light_Aircraft_Below_10000_ft_Data.mat`
- Number of tracks: 10
- Simulation duration: 250 seconds per track
- Python version: 3.12.10
- Platform: macOS (Apple Silicon)

## Baseline Benchmark Results

| Metric | Value |
|--------|-------|
| Mean time (10 tracks) | 1.35s |
| Throughput | 7.39 tracks/second |
| Time per track | ~135ms |
| Std deviation | 0.11s |

## Component Timing Breakdown (Single Track)

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Initial sampling | 0.09 | 0.1% |
| Transition sampling | 7.82 | 11.0% |
| Track simulation | 63.42 | 88.9% |
| **Total** | **71.32** | **100%** |

## Profiler Hot Spots

Based on profiling 10 tracks × 250s simulation:

| Function | Calls | Cumulative Time (s) | Per Call (ms) |
|----------|-------|---------------------|---------------|
| `dynamics.py:simulate_track` | 32 | 1.557 | 48.67 |
| `dynamics.py:integrate_single_time_step` | 80,000 | 1.303 | 0.016 |
| `bayesian_network.py:_sample_dependent_variable_transitions` | 32 | 0.477 | 14.91 |
| `data_classes.py:compute_trigonometric_values` | 80,000 | 0.426 | 0.005 |
| `data_classes.py:from_euler_angles` | 80,000 | 0.403 | 0.005 |
| `dynamics.py:calculate_maximum_allowable_bank_angle` | 80,000 | 0.234 | 0.003 |
| `dynamics.py:calculate_pitch_and_yaw_rates` | 80,000 | 0.162 | 0.002 |
| `utilities.py:calculate_conditional_probability_table_index` | 24,160 | 0.156 | 0.006 |
| `utilities.py:saturate_value_within_limits` | 320,000 | 0.142 | 0.0004 |

## Key Observations

### Primary Bottleneck: Track Simulation (~89% of time)
The dynamics integration loop in `simulate_track` dominates execution time:
- 80,000 calls to `integrate_single_time_step` per benchmark run
- Each call involves trigonometric calculations and state updates
- `compute_trigonometric_values` is called for every integration step

### Secondary Bottleneck: Transition Sampling (~11% of time)
- `_sample_dependent_variable_transitions` is called 32 times
- Significant time spent in conditional probability table indexing

### Optimization Opportunities

1. **Trigonometric Calculations**
   - `TrigonometricStateValues.from_euler_angles` is called 80,000 times
   - Consider caching or vectorizing trigonometric operations
   - sin, cos, tan computed redundantly

2. **Dynamics Integration Loop**
   - Loop in `simulate_track` has overhead per iteration
   - Consider NumPy vectorization for state updates
   - Pre-allocate arrays and use in-place operations

3. **Probability Sampling**
   - `calculate_conditional_probability_table_index` has loop overhead
   - Can be vectorized using NumPy operations

4. **Saturation Functions**
   - 320,000 calls to `saturate_value_within_limits`
   - Use `np.clip` instead of custom function

5. **State Object Creation**
   - Each integration step creates new `AircraftKinematicState` object
   - Consider mutable state or array-based approach

## Next Steps

1. Optimize trigonometric calculations by caching/vectorizing
2. Vectorize the dynamics integration loop where possible
3. Use NumPy built-ins instead of custom Python functions
4. Reduce object creation overhead in hot paths
5. Profile after each optimization to measure improvement
