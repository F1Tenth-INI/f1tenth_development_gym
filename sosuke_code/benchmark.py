import time
import numpy as np
from numba import njit

@njit
def compute_heavy(x):
    for i in range(x.shape[0]):
        x[i] = np.sin(x[i]) + np.log1p(x[i])
    return x

# Benchmark function
def benchmark():
    x = np.linspace(0.1, 100, 10**6)
    
    # Warm-up (triggers JIT compilation)
    compute_heavy(x.copy())

    # Measure execution time
    start = time.perf_counter()
    compute_heavy(x.copy())
    end = time.perf_counter()

    print(f"Execution time: {end - start:.6f} seconds")

if __name__ == "__main__":
    benchmark()
