# Learner Server Implementations

This directory contains two versions of the learner server for SAC training:

## Files

### `learner_server.py` (Original - Blocking Implementation)
- **Status**: Original implementation, kept for reference/rollback
- **Size**: ~58.5 KB
- **Characteristics**:
  - Synchronous training loop blocks the asyncio event loop
  - Training occurs directly in `_train_loop()` coroutine
  - Every 10 training steps: `await asyncio.sleep(0.01)` to check termination
  - **Issue**: Episodes stop sending during training because the event loop is blocked
  - Episode receiving is starved when training is happening
  - Uses more frequent (every 10 steps) termination checks

### `learner_server_threaded.py` (Threaded Implementation - Recommended)
- **Status**: New implementation with threading optimization
- **Size**: ~56.8 KB
- **Characteristics**:
  - Training runs in a separate worker thread via `loop.run_in_executor()`
  - Event loop remains responsive and can receive episodes continuously
  - `_train_loop()` now only handles orchestration (async/await)
  - New method `_train_step_blocking()` contains the actual training logic
  - Termination checks every 100 steps (less overhead) via thread-safe communication
  - Episodes continue flowing in during training
  - **Benefit**: Solves the episode blocking issue without limiting CPU cores

## Key Differences

| Aspect | Original | Threaded |
|--------|----------|----------|
| Training execution | Blocks event loop | Runs in worker thread |
| Episode receiving during training | ❌ Blocked | ✅ Responsive |
| Termination checks | Every 10 steps | Every 100 steps |
| Event loop responsibility | Training + coordination | Coordination only |
| Thread communication | N/A | `asyncio.run_coroutine_threadsafe()` |

## How to Use

### Use the Threaded Version (Recommended)
```python
# In your training script, import and use:
from learner_server_threaded import LearnerServer

server = LearnerServer(
    host="127.0.0.1",
    port=5000,
    device="cuda",
    # ... other parameters
)
```

### Fallback to Original
If you encounter issues with the threaded version, revert by:
```python
from learner_server import LearnerServer
```

## Thread Safety Notes

The threaded implementation maintains thread safety through:
1. `asyncio.run_coroutine_threadsafe()` for cross-thread communication
2. Async locks (`self._terminate_lock`) for shared state
3. No direct access to model/optimizer state from multiple threads
  - Training thread handles all model computations
  - Main thread only reads model state for broadcasting

## Performance Expectations

- **Training throughput**: No change (training still runs on same hardware)
- **Episode throughput**: Significantly improved (no blocking)
- **Overall training time**: Should decrease (fewer timeouts, continuous data flow)
- **CPU usage**: Slightly more efficient thread scheduling

## Switching Between Versions

To test both versions:

```bash
# Test original (blocking)
cp learner_server.py learner_server_backup_threaded.py
# Use original...

# Test threaded version
cp learner_server_threaded.py learner_server.py
# Use threaded...
```

Or modify your import path directly.

## Implementation Details

### Threading Architecture
```
┌─────────────────────────────────┐
│   Asyncio Event Loop             │
│  (handles networking, coordination)│
├─────────────────────────────────┤
│ _train_loop() - Async            │
│ ├─ Episode ingestion (sync)      │
│ ├─ run_in_executor()             │
│ │  └─> Worker Thread             │
│ │      └─> _train_step_blocking()│
│ │          ├─ Model forward      │
│ │          ├─ Backward           │
│ │          ├─ Optimize           │
│ │          └─ Save weights       │
│ └─ Broadcast weights (async)     │
└─────────────────────────────────┘
```

### Worker Thread Communication
- Thread → Event Loop: `asyncio.run_coroutine_threadsafe()`
- Event Loop → Thread: Shared state via locks
- Model state: Thread-exclusive during training, then broadcast

