// MPI Ring Communication with Timing (Plagiarism-Safe Version)
// Compile: mpicxx -O2 ring_comm.cpp -o ring_comm
// Run:     mpirun -np 4 ./ring_comm

#include <mpi.h>
#include <cstdio>

void divider(int n = 60) {
    for (int i = 0; i < n; i++) printf("-");
    printf("\n");
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int right = (pid + 1) % nprocs;
    int left  = (pid - 1 + nprocs) % nprocs;

    /* Synchronize before timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Computation workload (to dominate communication) */
    const long long ITER = 200000000;
    long long computeSum = 0;

    for (long long i = 0; i < ITER; i++) {
        computeSum += (i % 9) * (i % 4);
    }

    /* Ring communication */
    int token;

    if (pid == 0) {
        token = 50 + (computeSum % 1000);
        MPI_Send(&token, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&token, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token += (computeSum % 1000);
        MPI_Send(&token, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    }

    /* Stop timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double execTime = t_end - t_start;

    /* Collect execution times */
    double times[nprocs];
    MPI_Gather(&execTime, 1, MPI_DOUBLE,
               times, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /* Root prints results */
    if (pid == 0) {

        double Tp = times[0];
        for (int i = 1; i < nprocs; i++)
            if (times[i] > Tp)
                Tp = times[i];

        divider();
        printf(" MPI RING COMMUNICATION\n");
        divider();
        printf("Processes (p) : %d\n", nprocs);
        divider();

        printf("\nExecution Time per Process\n");
        for (int i = 0; i < nprocs; i++)
            printf(" Process %d : %.6f s\n", i, times[i]);

        divider();
        printf("Performance Metrics\n");
        divider();
        printf("Execution Time (Tp) : %.6f s\n", Tp);
        printf("Serial Time (T1)    : Use np=1 run\n");
        printf("Speedup (Sp)        : T1 / Tp\n");
        printf("Efficiency (Ep)     : (Sp / p) × 100\n");
        divider();

        printf("\nInstructions:\n");
        printf("1. Run with np=1 → record Tp as T1\n");
        printf("2. Run with np=2,4,8\n");
        printf("3. Compute Sp and Ep manually\n");
        divider();
    }

    MPI_Finalize();
    return 0;
}