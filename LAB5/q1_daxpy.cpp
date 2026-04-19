#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  long long n = 1LL << 16;
  if (argc > 1) {
    n = std::atoll(argv[1]);
  }

  const double a = 2.5;

  const long long base = n / size;
  const long long rem = n % size;
  const long long local_n = base + (rank < rem ? 1 : 0);

  std::vector<double> x(local_n, 1.0);
  std::vector<double> y(local_n, 2.0);

  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  for (long long i = 0; i < local_n; ++i) {
    x[i] = a * x[i] + y[i];
  }
  const double end = MPI_Wtime();

  const double local_time = end - start;
  double max_time = 0.0;
  MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "DAXPY size=" << n << " time=" << max_time << " s" << std::endl;
  }

  MPI_Finalize();
  return 0;
}