// MPI Program: Global Minimum & Maximum with Performance Evaluation
// Compile: mpicxx -O2 array_minmax.cpp -o array_minmax
// Run:     mpirun -np 4 ./array_minmax

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

void drawLine(int width = 70) {
    for (int i = 0; i < width; i++) printf("-");
    printf("\n");
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int LOCAL_N = 10;
    int data[LOCAL_N];

    srand(time(nullptr) + pid);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Generate random data */
    for (int i = 0; i < LOCAL_N; i++)
        data[i] = rand() % 1001;

    /* Local min and max */
    int localMin = data[0];
    int localMax = data[0];

    for (int i = 1; i < LOCAL_N; i++) {
        if (data[i] < localMin) localMin = data[i];
        if (data[i] > localMax) localMax = data[i];
    }

    /* Structures for location-aware reduction */
    struct {
        int val;
        int proc;
    } maxPair, minPair, gMaxPair, gMinPair;

    maxPair.val  = localMax;
    maxPair.proc = pid;

    minPair.val  = localMin;
    minPair.proc = pid;

    /* Global min/max with process ID */
    MPI_Reduce(&maxPair, &gMaxPair, 1,
               MPI_2INT, MPI_MAXLOC,
               0, MPI_COMM_WORLD);

    MPI_Reduce(&minPair, &gMinPair, 1,
               MPI_2INT, MPI_MINLOC,
               0, MPI_COMM_WORLD);

    /* Plain reductions */
    int globalMax, globalMin;

    MPI_Reduce(&localMax, &globalMax,
               1, MPI_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    MPI_Reduce(&localMin, &globalMin,
               1, MPI_INT, MPI_MIN,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double execTime = t_end - t_start;
    double Tp = 0.0;

    MPI_Reduce(&execTime, &Tp, 1,
               MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (pid == 0) {

        static double T1 = 0.0;
        if (nprocs == 1)
            T1 = Tp;

        double speedup = 0.0;
        double efficiency = 0.0;
        double overhead = 0.0;

        if (T1 > 0) {
            speedup = T1 / Tp;
            efficiency = (speedup / nprocs) * 100.0;

            double ideal = T1 / nprocs;
            overhead = ((Tp - ideal) / Tp) * 100.0;
        }

        drawLine();
        printf(" MPI GLOBAL MINIMUM & MAXIMUM\n");
        drawLine();
        printf("Processes            : %d\n", nprocs);
        printf("Elements per process : %d\n", LOCAL_N);
        printf("Total elements       : %d\n", LOCAL_N * nprocs);
        drawLine();

        printf("Global Maximum       : %d\n", globalMax);
        printf("Max found by process : %d\n", gMaxPair.proc);

        printf("\nGlobal Minimum       : %d\n", globalMin);
        printf("Min found by process : %d\n", gMinPair.proc);
        drawLine();

        printf("Performance Metrics\n");
        drawLine();
        printf("Parallel Time (Tp)   : %.8f s\n", Tp);

        if (nprocs == 1)
            printf("Serial Time (T1)     : %.8f s\n", Tp);
        else
            printf("Serial Time (T1)     : Use np=1 value\n");

        printf("Speedup              : %.4f\n", speedup);
        printf("Efficiency           : %.2f %%\n", efficiency);
        printf("Communication Overhead: %.2f %%\n", overhead);
        drawLine();
    }

    MPI_Finalize();
    return 0;
}