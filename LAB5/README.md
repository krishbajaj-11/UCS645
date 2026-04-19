# MPI Assignment 5 Report

# Overview

This report summarizes five MPI exercises that explore communication patterns and work partitioning. The focus is on collective operations, point to point messaging, and master worker scheduling.

The tasks covered are: DAXPY vector update, manual broadcast versus MPI_Bcast, distributed dot product with MPI_Bcast and MPI_Reduce, and dynamic work distribution for prime and perfect number detection.

---

# System Configuration

All runs were performed on Linux using the MPI C++ compiler mpicxx. Process counts of 1, 2, 4, 8, and 16 were used where timing data was collected.

---

# Question 1 DAXPY Operation

## Description

DAXPY performs the vector update
X[i] = a multiplied by X[i] plus Y[i]

The MPI version splits the work across ranks and reports time and speedup relative to a single process run.

## Execution Time (Parallel Time)

| Number of Processes | Time in seconds |
| ------------------- | --------------- |
| 1                   | 0.000033971     |
| 2                   | 0.000081575     |
| 4                   | 0.000082238     |
| 8                   | 0.000096656     |
| 16                  | 0.000090401     |

## Speedup Calculation

Speedup is computed as $T_1 / T_N$.

| Number of Processes | Speedup |
| ------------------- | ------- |
| 2                   | 0.416   |
| 4                   | 0.413   |
| 8                   | 0.351   |
| 16                  | 0.376   |

## Speedup Graph

![Q1 Speedup vs Processes](images/q1_speedup.png)

## Detailed Analysis

Runtime does not improve on this machine because the workload is tiny and the launch and synchronization overheads dominate the compute loop.

---

# Question 2 Broadcast Race

## Description

This task contrasts a manual broadcast implemented with repeated MPI_Send calls against MPI_Bcast, which uses an optimized collective algorithm.

## Execution Time

| Number of Processes | Manual Broadcast Time | MPI Broadcast Time |
| ------------------- | --------------------- | ------------------ |
| 2                   | 0.000000500           | 0.000000463        |
| 4                   | 0.000000762           | 0.000000824        |
| 8                   | 0.000000940           | 0.000000858        |
| 16                  | 0.000005717           | 0.000000752        |

## Speedup Graph (MPI_Bcast vs MyBcast)

Speedup is computed as MyBcast time divided by MPI_Bcast time.

| Number of Processes | Speedup Ratio |
| ------------------- | ------------- |
| 2                   | 1.080         |
| 4                   | 0.925         |
| 8                   | 1.096         |
| 16                  | 7.602         |

![Q2 Speedup: MPI_Bcast vs MyBcast](images/q2_speedup.png)

## Detailed Analysis

The manual approach makes rank 0 a bottleneck because it must send to every rank sequentially. MPI_Bcast avoids this linear bottleneck by forwarding data in a tree, which reduces the number of steps as process count grows.

---

# Question 3 Distributed Dot Product and Amdahl Law

## Description

Each process generates its local chunk of two vectors, computes a partial dot product, and MPI_Reduce combines the partial sums. A multiplier is broadcast from rank 0 before the local computation starts.

## Execution Time

| Number of Processes | Time in seconds |
| ------------------- | --------------- |
| 1                   | 0.431899        |
| 2                   | 0.469746        |
| 4                   | 0.576816        |
| 8                   | 0.887827        |

## Speedup

| Number of Processes | Speedup |
| ------------------- | ------- |
| 2                   | 0.919   |
| 4                   | 0.749   |
| 8                   | 0.486   |

## Speedup Graph

![Q3 Speedup vs Processes](images/q3_speedup.png)

## Efficiency

| Number of Processes | Efficiency |
| ------------------- | ---------- |
| 2                   | 0.460      |
| 4                   | 0.187      |
| 8                   | 0.061      |

## Detailed Analysis

Speedup is below 1.0 on this machine because the broadcast and reduction overheads dominate relative to the local compute work. This matches the Amdahl Law expectation that non parallel work can limit overall scaling.

---

# Question 4 Prime Number Computation

## Description

The prime finder uses a master worker scheme where workers request candidates, test for primality, and return the result to the master.

## Output

The program correctly identifies all prime numbers up to 100.

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97


## Detailed Analysis

Dynamic task assignment keeps workers busy and avoids idle time that can happen with static chunking, especially when per number work is uneven.

---

# Question 5 Perfect Number Computation

## Description

This task mirrors the prime worker model but checks whether a number equals the sum of its proper divisors.

## Output

The program identifies the following perfect numbers up to 10000

6, 28, 496, 8128


## Detailed Analysis

Checking perfect numbers is heavier than primality testing, and the master worker pattern helps distribute that cost evenly. Since valid results are rare, the output is small but the computation is significant.

---

# Final Conclusion

These experiments show the benefit of MPI parallelism alongside its limitations. Collective operations like MPI_Bcast scale better than naive point to point loops, and master worker scheduling improves utilization for irregular workloads.

Even with good parallel structure, synchronization and communication overheads cap the achievable speedup, which is consistent with Amdahl Law.