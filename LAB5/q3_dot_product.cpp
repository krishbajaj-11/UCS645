#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  long long total_n = 500000000LL;
  if (argc > 1) {
    total_n = std::atoll(argv[1]);
  }

  double multiplier = 1.0;
  if (rank == 0 && argc > 2) {
    multiplier = std::atof(argv[2]);
  }

  MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const long long base = total_n / size;
  const long long rem = total_n % size;
  const long long local_n = base + (rank < rem ? 1 : 0);
  const long long start_idx = rank * base + (rank < rem ? rank : rem);

  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  double local_sum = 0.0;
  for (long long i = 0; i < local_n; ++i) {
    const long long global_i = start_idx + i;
    const double a = 1.0 + (global_i % 3) * 0.0;
    const double b = 2.0 * multiplier;
    local_sum += a * b;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  const double end = MPI_Wtime();
  const double local_time = end - start;
  double max_time = 0.0;
  MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Dot size=" << total_n << " multiplier=" << multiplier
              << " result=" << global_sum << " time=" << max_time << " s"
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}