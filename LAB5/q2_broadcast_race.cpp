#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

void my_bcast(std::vector<double>& buffer, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == root) {
    for (int dest = 0; dest < size; ++dest) {
      if (dest == root) {
        continue;
      }
      MPI_Send(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE,
               dest, 0, comm);
    }
  } else {
    MPI_Recv(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE,
             root, 0, comm, MPI_STATUS_IGNORE);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  long long n = 10000000;
  if (argc > 1) {
    n = std::atoll(argv[1]);
  }

  std::vector<double> buffer(n, 0.0);
  if (rank == 0) {
    for (long long i = 0; i < n; ++i) {
      buffer[i] = 1.0;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  my_bcast(buffer, 0, MPI_COMM_WORLD);
  double end = MPI_Wtime();
  double local_time = end - start;

  double max_time = 0.0;
  MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "MyBcast size=" << n << " time=" << max_time << " s" << std::endl;
  }

  if (rank == 0) {
    for (long long i = 0; i < n; ++i) {
      buffer[i] = 2.0;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
  MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
  end = MPI_Wtime();
  local_time = end - start;

  max_time = 0.0;
  MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "MPI_Bcast size=" << n << " time=" << max_time << " s" << std::endl;
  }

  MPI_Finalize();
  return 0;
}