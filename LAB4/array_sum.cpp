// Parallel Sum of Array using MPI with Performance Analysis
// Compile: mpicxx -O2 -o mpi_array_sum mpi_array_sum.cpp
// Run:     mpirun -np 4 ./mpi_array_sum

#include <mpi.h>
#include <cstdio>
#include <cstdlib>

void printLine(int len = 70) {
    while (len--) printf("-");
    printf("\n");
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = 100;
    int *globalData = nullptr;

    /* Determine data distribution */
    int chunk = N / nprocs;
    int extra = N % nprocs;

    int myCount = chunk + (pid < extra ? 1 : 0);
    int *myData = (int*)malloc(myCount * sizeof(int));

    int *counts = nullptr;
    int *offsets = nullptr;

    if (pid == 0) {
        globalData = (int*)malloc(N * sizeof(int));

        for (int i = 0; i < N; i++)
            globalData[i] = i + 1;

        counts  = (int*)malloc(nprocs * sizeof(int));
        offsets = (int*)malloc(nprocs * sizeof(int));

        int pos = 0;
        for (int i = 0; i < nprocs; i++) {
            counts[i]  = chunk + (i < extra ? 1 : 0);
            offsets[i] = pos;
            pos += counts[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    /* Scatter data */
    MPI_Scatterv(globalData, counts, offsets,
                 MPI_INT,
                 myData, myCount,
                 MPI_INT,
                 0, MPI_COMM_WORLD);

    /* Compute partial sum */
    int partialSum = 0;
    for (int i = 0; i < myCount; i++)
        partialSum += myData[i];

    int totalSum = 0;
    MPI_Reduce(&partialSum, &totalSum,
               1, MPI_INT,
               MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double tEnd = MPI_Wtime();

    double localTime = tEnd - tStart;
    double parallelTime = 0;

    MPI_Reduce(&localTime, &parallelTime,
               1, MPI_DOUBLE,
               MPI_MAX,
               0, MPI_COMM_WORLD);

    if (pid == 0) {

        static double serialTime = 0.0;

        if (nprocs == 1)
            serialTime = parallelTime;

        double speedup = 0.0;
        double efficiency = 0.0;
        double commOverhead = 0.0;

        if (serialTime > 0) {
            speedup = serialTime / parallelTime;
            efficiency = (speedup / nprocs) * 100.0;

            double idealTime = serialTime / nprocs;
            commOverhead = ((parallelTime - idealTime) / parallelTime) * 100.0;
        }

        printLine();
        printf(" MPI PARALLEL ARRAY SUM\n");
        printLine();
        printf("Processes           : %d\n", nprocs);
        printf("Array Elements      : %d\n", N);
        printf("Computed Sum        : %d\n", totalSum);
        printf("Expected Sum        : 5050\n");
        printf("Mean Value          : %.2f\n", (double)totalSum / N);
        printf("Validation          : %s\n",
               totalSum == 5050 ? "SUCCESS ✓" : "FAILED ✗");
        printLine();

        printf("Performance Analysis\n");
        printLine();
        printf("Parallel Time (Tp)  : %.8f s\n", parallelTime);

        if (nprocs == 1)
            printf("Serial Time (T1)    : %.8f s\n", parallelTime);
        else
            printf("Serial Time (T1)    : From np=1 execution\n");

        printf("Speedup (Sp)        : %.4f\n", speedup);
        printf("Efficiency (Ep)     : %.2f %%\n", efficiency);
        printf("Comm Overhead       : %.2f %%\n", commOverhead);
        printLine();

        printf("\nTABLE FORMAT\n");
        printf("%-5d %-12.8f %-10.4f %-10.2f %-10.2f\n",
               nprocs, parallelTime, speedup, efficiency, commOverhead);
        printLine();
    }

    MPI_Finalize();
    return 0;
}