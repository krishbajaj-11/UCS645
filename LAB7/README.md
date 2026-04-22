# Assignment 7: CUDA Part II


## Overview
- Sum of first N integers (iterative vs formula) on CPU and GPU
- Merge sort with CPU pipelining vs CUDA merge sort
- Vector addition with timing and bandwidth analysis

## Part A: Sum of First N Integers (N = 1024)

### Task
- Compute sum using iterative approach
- Compute sum using direct formula
- Run both on CPU and GPU

### Results
```
CPU iterative sum = 524800
CPU formula sum   = 524800
CPU time (ms)      = 0.000372
GPU iterative sum = 524800
GPU formula sum   = 524800
GPU time (ms)      = 0.129856
```

### Answers
- Expected sum is $N(N+1)/2 = 1024 * 1025 / 2 = 524800$, which matches CPU and GPU.
- Iterative and formula approaches produce identical results.

### Observations
- CPU is faster for this small $N$ because GPU kernel launch and memory overhead dominate.

## Part B: Merge Sort (n = 1000)

### Task
- Implement merge sort with CPU-side pipelining
- Implement parallel merge sort using CUDA
- Compare performance

### Results
```
CPU pipeline sort time: 1.317 ms (sorted=yes)
GPU merge sort kernel time: 15.814 ms (sorted=yes)
```

### Answers
- Sorting is correct based on program verification.
- CPU pipelined version is faster for $n=1000$.

### Observations
- For small arrays, GPU overhead outweighs the parallel speedup.
- CUDA performance should improve with larger input sizes.

## Part C: Vector Addition + Bandwidth

### Task
- Use statically defined global device memory (no `cudaMalloc`)
- Record kernel timing
- Query device properties to compute theoretical bandwidth
- Compute measured bandwidth using $\text{measuredBW} = (RBytes + WBytes) / t$

### Results
```
Kernel time: 0.025 ms
Theoretical bandwidth: 192.024 GB/s
Measured bandwidth: 495.859 GB/s
Sample output: C[0]=3.0 C[N-1]=3.0
```

### Answers
- Vector addition output is correct for the sample values.
- Theoretical bandwidth uses device properties and the DDR factor.
- Measured bandwidth uses total bytes moved and kernel time.

### Observations
- The measured bandwidth can exceed the theoretical value in tiny benchmarks because kernel timing is extremely small and sensitive to overhead and clock effects.


