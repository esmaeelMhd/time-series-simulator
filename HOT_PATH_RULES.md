# Python Hot-Path Performance Rules

These rules apply only to **performance-critical (hot) paths**.  
Always profile first and identify the hot path before applying them.

---

## Rule 1: Identify the hot path before optimizing

- Never guess performance bottlenecks.
- Use profiling tools (`cProfile`, `line_profiler`, `perf`, PyTorch profiler, etc.).
- Apply these rules only where runtime is dominated.

---

## Rule 2: No Python loops over large data in the hot path

- Avoid `for`, `while`, comprehensions, and `map` on large datasets.
- Replace with:
  - Vectorized NumPy operations
  - Library kernels (BLAS, LAPACK, Torch ops)
  - JIT-compiled code (Numba, Cython)
- Small loops around large vectorized operations are acceptable.

**Example - Before:**
```python
# BAD: Python loop over batch
for i in range(batch_size):
    result[i] = compute(data[i])
```

**Example - After:**
```python
# GOOD: Vectorized batch operation
result = compute_batch(data)  # Single call processes all items
```

---

## Rule 3: Avoid Python objects in the hot path

- Do not use Python `list`, `dict`, `tuple`, or custom objects.
- Use:
  - Typed, contiguous arrays (`numpy.ndarray`, tensors, memoryviews)
  - Fixed dtypes and shapes
- Python objects are allowed at API boundaries, not inside kernels.

---

## Rule 4: Minimize Python call overhead

- Avoid repeated Python function calls in tight loops.
- Prefer:
  - Batching
  - Operation fusion
  - Single large calls instead of many small ones
- Let libraries execute large kernels internally.

**Example - Before:**
```python
# BAD: Repeated small calls
for t in range(horizon):
    pred = model.step(state, input[t])
    predictions.append(pred)
predictions = torch.stack(predictions)
```

**Example - After:**
```python
# GOOD: Single batched call when possible
predictions = model.forward_batch(state, inputs)  # Processes all timesteps
```

---

## Rule 5: No memory allocations in the hot path

- Do not allocate arrays, lists, or tensors inside critical loops.
- Preallocate buffers and reuse them.
- Avoid temporary objects and implicit copies.

**Example - Before:**
```python
# BAD: Allocations inside loop
predictions = []
for t in range(horizon):
    pred = model.step(...)
    predictions.append(pred)  # List grows dynamically
predictions = torch.stack(predictions)  # Creates new tensor
```

**Example - After:**
```python
# GOOD: Preallocated output tensor
predictions = torch.empty(batch_size, horizon, output_dim, device=device)
for t in range(horizon):
    predictions[:, t, :] = model.step(...)  # Direct assignment
```

---

## Rule 6: Avoid data-dependent branching in the hot path

- Avoid `if/else` on per-element data.
- Prefer:
  - Boolean masks
  - `where` / lookup tables
  - Vectorized condition handling
- Predictable, loop-level branches are acceptable.

**Example - Before:**
```python
# BAD: Per-element branching
for i in range(n):
    if data[i] > threshold:
        result[i] = process_high(data[i])
    else:
        result[i] = process_low(data[i])
```

**Example - After:**
```python
# GOOD: Vectorized with mask
mask = data > threshold
result = torch.where(mask, process_high(data), process_low(data))
```

---

## Rule 7: Avoid data conversions in the hot path

- No:
  - `list` ↔ `ndarray` conversions
  - dtype casts (`astype`)
  - shape reshaping that triggers copies
- Choose dtype, layout, and device once, early.
- Stay in that representation throughout the hot path.

**Example - Before:**
```python
# BAD: Repeated conversions inside loop
for k in range(batch_size):
    D_k = D[k].detach().cpu().numpy()      # GPU → CPU
    R_k = compute(D_k)
    R[k] = torch.FloatTensor(R_k).to(dev)  # CPU → GPU
```

**Example - After:**
```python
# GOOD: Single batch conversion
D_cpu = D.detach().cpu().numpy()           # One transfer
R_cpu = np.zeros((batch_size, N, N))       # Preallocate
for k in range(batch_size):
    R_cpu[k] = compute(D_cpu[k])           # Pure numpy operations
R = torch.from_numpy(R_cpu).to(device)     # One transfer back
```

---

## Rule 8: Avoid I/O and logging in the hot path

- No printing, file I/O, network calls, or logging.
- Collect metrics in memory and report them outside the hot path.
- Sampling or conditional logging must be extremely rare.

---

## Rule 9: Respect memory layout and locality

- Use contiguous arrays whenever possible.
- Match access patterns to layout (C-order vs Fortran-order).
- Favor linear access over strided or random access.
- Memory bandwidth is often the real bottleneck.

---

## Rule 10: Prefer libraries that generate optimized kernels

- Use libraries that move computation out of Python:
  - NumPy ufuncs
  - BLAS/LAPACK
  - PyTorch / JAX
  - Numba
- Python should orchestrate, not compute.

---

## Rule 11: Optimize for clarity outside the hot path

- Readability, correctness, and maintainability come first elsewhere.
- Only sacrifice clarity where performance is proven critical.
- Clearly document why a section is optimized.

---

## Hot Paths in This Repository

The following areas have been identified as hot paths and optimized:

| Module | Function | Optimization Applied |
|--------|----------|---------------------|
| `timesim/training/rollout.py` | `batch_rollout()` | Batched data preparation, uniform horizon fast path |
| `timesim/training/rollout.py` | `batch_rollout_padded()` | Vectorized mask creation |
| `timesim/data/sampling.py` | `RandomStartRandomHorizon.sample()` | Vectorized start_indices |
| `timesim/data/sampling.py` | `GeometricHorizonSampling.sample()` | Vectorized start_indices |
| `timesim/models/lstm.py` | `LSTMWorldModel.rollout()` | Preallocated predictions tensor |
| `timesim/models/base.py` | `WorldModelBase.rollout()` | Preallocated predictions tensor |
| `utils/dilate_loss.py` | `dilate_loss()` | Batched pairwise distances, cached Omega |
| `utils/soft_dtw.py` | `SoftDTWBatch` | Batched CPU/GPU transfers |
| `utils/path_soft_dtw.py` | `PathDTWBatch` | Batched CPU/GPU transfers |

---

## Profiling Commands

```bash
# CPU profiling with cProfile
python -m cProfile -s cumulative train.py

# Line-by-line profiling (install: pip install line_profiler)
kernprof -l -v train.py

# PyTorch profiler
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_step()
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Code Markers

Hot path sections are marked with `HOT PATH:` comments in the docstrings:

```python
def batch_rollout(...):
    """Perform batched rollouts.
    
    HOT PATH: This function is called every training step.
    Optimizations:
    - Batch data preparation with single numpy->tensor conversion
    - True batched rollout when horizons are uniform
    """
```

Search for these markers:
```bash
grep -r "HOT PATH" timesim/ utils/
```

