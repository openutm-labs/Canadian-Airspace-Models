# Performance Results - Round 2

## Date: 2026-01-29

## Optimizations Applied (Cumulative)

### Round 1 Optimizations (Kept)
1. **if/else saturation** - Faster than min/max for scalars
2. **math module for trig** - Faster than numpy for scalars

### Round 2 Optimization: Inline Dynamics Calculations
- **Change**: Completely inlined `integrate_single_time_step` method
- **Eliminated**: 7 method calls per iteration (560,000 calls total)
- **Cached**: Frequently accessed constants in `__init__`
- **Files modified**: [dynamics.py](../../../src/cam_track_gen/dynamics.py)

## Benchmark Results

| Metric | Baseline | Round 1 | Round 2 | Total Improvement |
|--------|----------|---------|---------|-------------------|
| Mean time (10 tracks) | 1.35s | 1.05s | 0.79s | **41.5%** |
| Throughput | 7.39 tracks/s | 9.51 tracks/s | 12.71 tracks/s | **72.0%** |
| Time per track | ~135ms | ~105ms | ~79ms | **41.5%** |

## New Profiler Hot Spots

| Function | Calls | Cumulative Time (s) | Per Call (ms) |
|----------|-------|---------------------|---------------|
| `dynamics.py:simulate_track` | 32 | 0.650 | 20.30 |
| `dynamics.py:integrate_single_time_step` | 80,000 | 0.423 | 0.005 |
| `bayesian_network.py:_sample_dependent_variable_transitions` | 32 | 0.399 | 12.46 |
| `utilities.py:calculate_conditional_probability_table_index` | 24,160 | 0.139 | 0.006 |
| `utilities.py:sample_from_distribution` | 24,192 | 0.105 | 0.004 |
| `bayesian_network.py:_process_transition_data_with_resampling` | 32 | 0.063 | 1.98 |
| `data_classes.py:to_output_array` | 80,032 | 0.056 | 0.001 |

## Time Distribution Analysis

| Component | Before Opt3 (s) | After Opt3 (s) | Reduction |
|-----------|-----------------|----------------|-----------|
| Dynamics integration | 0.935 | 0.423 | **55%** |
| Transition sampling | 0.393 | 0.399 | ~0% (unchanged) |
| Total execution | 1.63 | 1.13 | **31%** |

## Key Insight

The dynamics integration is now only ~37% of execution time (down from ~57%).
The Bayesian network sampling is now a bigger proportion (~35%).

## Next Optimization Opportunities

1. **Transition Sampling** (35% of time)
   - `_sample_dependent_variable_transitions` still has Python loop
   - `calculate_conditional_probability_table_index` called 24,160 times
   - Consider pre-computing or vectorizing

2. **Array Creation** (~10% of time)
   - `to_output_array` creates numpy arrays 80,000 times
   - Consider pre-allocating output buffer

3. **Simulation Loop** (~57% of time in simulate_track)
   - Loop still has per-iteration overhead
   - Consider state as arrays instead of objects
