# Round 4 Optimization Results

## Summary

| Metric | Baseline | Round 3 | Round 4 | Change from Baseline |
| ------ | -------- | ------- | ------- | -------------------- |
| Mean Time | 1.35s | 0.60s | 0.50s | **-62.8%** |
| Throughput | 7.39 tracks/s | 16.67 tracks/s | 19.92 tracks/s | **+169.6%** |
| Speedup | 1.0x | 2.25x | **2.70x** | |

## Optimizations Implemented

### Optimization 5: Pure Python Sampling Loop

Replaced NumPy cumsum/searchsorted with pure Python linear scan for small probability arrays.

**utilities.py changes:**

```python
# Before: NumPy cumsum + searchsorted
cumsum = np.cumsum(probability_weights)
threshold = cumsum[-1] * np.random.rand()
return int(np.searchsorted(cumsum, threshold))

# After: Pure Python loop (faster for small arrays of 2-10 elements)
total = 0.0
for w in probability_weights:
    total += w
threshold = total * np.random.rand()

cumulative = 0.0
for i, w in enumerate(probability_weights):
    cumulative += w
    if cumulative > threshold:
        return i
return len(probability_weights) - 1
```

**Rationale:** NumPy has significant overhead for small arrays. For typical probability tables (2-10 bins), pure Python is faster.

### Optimization 6: Eliminate to_output_array Calls

Removed 80,000 calls to `to_output_array` which created numpy arrays each time.

**dynamics.py changes:**

```python
# Before: Called to_output_array 80k times
state_history_buffer[step_index, :] = current_state.to_output_array(current_time)

# After: Direct buffer writes (no array allocation)
state_history_buffer[step_index, 0] = current_time
state_history_buffer[step_index, 1] = current_state.north_position_feet
state_history_buffer[step_index, 2] = current_state.east_position_feet
state_history_buffer[step_index, 3] = current_state.altitude_feet
state_history_buffer[step_index, 4] = current_state.velocity_feet_per_second
state_history_buffer[step_index, 5] = current_state.bank_angle_radians
state_history_buffer[step_index, 6] = current_state.pitch_angle_radians
state_history_buffer[step_index, 7] = current_state.heading_angle_radians
```

**Impact:** Eliminated 80,000 numpy array allocations per batch.

### Optimization 7: Replace np.where with Dict Lookup

Changed from `np.where` search (O(n) per lookup) to dict-based lookup (O(1)).

**dynamics.py changes:**

```python
# Before: np.where called 80k times
matching_time_indices = np.where(control_command_sequence[:, 0] == current_time)[0]
if matching_time_indices.size > 0:
    current_command_index = matching_time_indices[0]

# After: Pre-built dict with O(1) lookup
command_times = control_command_sequence[:, 0]
time_to_command_index: dict[float, int] = {}
for i, t in enumerate(command_times):
    time_to_command_index[float(t)] = i

# In loop:
if current_time in time_to_command_index:
    current_command_index = time_to_command_index[current_time]
```

**Impact:** Reduced command lookup from O(n×m) to O(n+m) where n=steps, m=commands.

## Benchmark Results

```
Configuration:
  Model: Light_Aircraft_Below_10000_ft_Data.mat
  Tracks per iteration: 10
  Simulation duration: 250s
  Benchmark iterations: 5

Results:
  Total tracks generated: 50
  Throughput: 19.92 tracks/second

Timing Statistics:
  Mean time: 0.5019s
  Std deviation: 0.0970s
  Min time: 0.4421s
  Max time: 0.6680s
  Median time: 0.4461s
```

## Progress Summary

| Round | Optimization | Time | Improvement |
| ----- | ------------ | ---- | ----------- |
| Baseline | - | 1.35s | - |
| Round 1 | if/else saturation, math module | 1.05s | 22% |
| Round 2 | Inlined dynamics | 0.79s | 42% |
| Round 3 | Cached parent indices | 0.60s | 56% |
| **Round 4** | **Pure Python sampling, direct buffer writes** | **0.50s** | **62.8%** |

## Test Verification

All 101 tests pass:

```
============================= 101 passed in 1.03s ==============================
```

## Next Round Assessment

### Profile After Round 4

Running profiler to identify remaining bottlenecks for Round 5...

### Expected Remaining Hotspots

1. **integrate_single_time_step** - Still the main loop, limited by Python overhead
2. **Trig calculations** - math.sin/cos called 240k times
3. **Object creation** - AircraftKinematicState created 80k times
4. **Random number generation** - numpy.random overhead

### Theoretical Limits

Current performance:

- 0.50s for 2,500,000 integration steps
- ~200ns per step (approaching Python's function call overhead limit)
- Further optimization would require Numba/Cython compilation or algorithm restructuring

### Potential Round 5 Optimizations

1. Cache trig values when angles don't change
2. Reduce object creation by using mutable state
3. Batch random number generation
4. Consider algorithmic changes to reduce integration steps
