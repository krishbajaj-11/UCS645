// MPI Program: Parallel Dot Product with Performance Evaluation
// Compile: mpicxx -O2 dot_product.cpp -o dot_product
// Run:     mpirun -np 4 ./dot_product

#include <mpi.h>
#include <cstdio>

void drawLine(int len = 70) {
    for (int i = 0; i < len; i++) printf("-");
    printf("\n");
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int SIZE = 8;

    /* Ensure equal distribution */
    if (SIZE % nprocs != 0) {
        if (pid == 0)
            printf("Vector length must be divisible by number of processes.\n");
        MPI_Finalize();
        return 0;
    }

    int block = SIZE / nprocs;

    int vecA[SIZE], vecB[SIZE];
    int subA[block], subB[block];

    /* Initialize vectors at root */
    if (pid == 0) {
        int initA[SIZE] = {1,2,3,4,5,6,7,8};
        int initB[SIZE] = {8,7,6,5,4,3,2,1};

        for (int i = 0; i < SIZE; i++) {
            vecA[i] = initA[i];
            vecB[i] = initB[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    /* Distribute vector segments */
    MPI_Scatter(vecA, block, MPI_INT,
                subA, block, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(vecB, block, MPI_INT,
                subB, block, MPI_INT,
                0, MPI_COMM_WORLD);

    /* Compute partial dot product */
    int partialDot = 0;
    for (int i = 0; i < block; i++)
        partialDot += subA[i] * subB[i];

    /* Reduce to global dot product */
    int finalDot = 0;
    MPI_Reduce(&partialDot, &finalDot,
               1, MPI_INT, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double tEnd = MPI_Wtime();

    double elapsed = tEnd - tStart;
    double Tp = 0.0;

    MPI_Reduce(&elapsed, &Tp,
               1, MPI_DOUBLE, MPI_MAX,
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
        printf(" MPI PARALLEL DOT PRODUCT\n");
        drawLine();
        printf("Vector Size (n)      : %d\n", SIZE);
        printf("Processes (p)        : %d\n", nprocs);
        drawLine();

        printf("Dot Product Result   : %d\n", finalDot);
        printf("Expected Result      : 120\n");
        printf("Validation           : %s\n",
               finalDot == 120 ? "SUCCESS ✓" : "FAILED ✗");
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