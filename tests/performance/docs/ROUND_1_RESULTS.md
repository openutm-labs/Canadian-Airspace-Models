# Performance Results - Round 1

## Date: 2026-01-29

## Optimizations Applied

### Optimization 1: Replace min/max with if/else in `saturate_value_within_limits`
- **Change**: Replaced `max(minimum_limit, min(maximum_limit, value))` with explicit if/else
- **Rationale**: Reduces function call overhead for scalar operations
- **Files modified**: [utilities.py](../../../src/cam_track_gen/utilities.py)

### Optimization 2: Use math module for trigonometric calculations
- **Change**: Replaced `np.sin()`, `np.cos()`, `np.tan()` with `math.sin()`, `math.cos()`
- **Rationale**: Python's math module is faster than NumPy for scalar operations
- **Also**: Compute tangent from sin/cos to avoid redundant calculation
- **Files modified**: [data_classes.py](../../../src/cam_track_gen/data_classes.py)

## Benchmark Results

| Metric | Baseline | After Opt 1 | After Opt 2 | Improvement |
|--------|----------|-------------|-------------|-------------|
| Mean time (10 tracks) | 1.35s | 1.26s | 1.05s | **22.2%** |
| Throughput | 7.39 tracks/s | 7.96 tracks/s | 9.51 tracks/s | **28.7%** |
| Time per track | ~135ms | ~126ms | ~105ms | **22.2%** |

## New Profiler Hot Spots

Based on profiling after optimizations:

| Function | Calls | Cumulative Time (s) | Per Call (ms) |
|----------|-------|---------------------|---------------|
| `dynamics.py:simulate_track` | 32 | 1.164 | 36.39 |
| `dynamics.py:integrate_single_time_step` | 80,000 | 0.935 | 0.012 |
| `bayesian_network.py:_sample_dependent_variable_transitions` | 32 | 0.393 | 12.28 |
| `dynamics.py:calculate_maximum_allowable_bank_angle` | 80,000 | 0.224 | 0.003 |
| `data_classes.py:compute_trigonometric_values` | 80,000 | 0.198 | 0.002 |
| `data_classes.py:from_euler_angles` | 80,000 | 0.177 | 0.002 |
| `utilities.py:calculate_conditional_probability_table_index` | 24,160 | 0.137 | 0.006 |
| `dynamics.py:_integrate_state_variables` | 80,000 | 0.135 | 0.002 |

## Time Reduction Analysis

| Component | Before (s) | After (s) | Reduction |
|-----------|-----------|-----------|-----------|
| Trigonometric calculations | 0.829 | 0.375 | **55%** |
| Total execution | 2.13 | 1.63 | **23%** |

## Next Optimization Opportunities

1. **Dynamics Integration Loop** (still ~57% of time)
   - `integrate_single_time_step` still the main bottleneck
   - Consider inlining calculations to reduce method call overhead
   - `calculate_maximum_allowable_bank_angle` has expensive sqrt operation

2. **Transition Sampling** (~24% of time)
   - `_sample_dependent_variable_transitions` loop can be optimized
   - `calculate_conditional_probability_table_index` called 24,160 times

3. **Object Creation Overhead**
   - `_integrate_state_variables` creates new `AircraftKinematicState` each step
   - `to_output_array` creates numpy array 80,000 times

4. **Array Operations**
   - Pre-allocate arrays in simulation loop
   - Avoid creating intermediate arrays
