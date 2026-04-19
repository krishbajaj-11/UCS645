#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static bool is_perfect(int n) {
  if (n < 2) {
    return false;
  }
  int sum = 1;
  const int limit = static_cast<int>(std::sqrt(static_cast<double>(n)));
  for (int d = 2; d <= limit; ++d) {
    if (n % d == 0) {
      sum += d;
      const int other = n / d;
      if (other != d) {
        sum += other;
      }
    }
  }
  return sum == n;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int max_n = 10000;
  if (argc > 1) {
    max_n = std::atoi(argv[1]);
  }

  if (size < 2) {
    if (rank == 0) {
      std::vector<int> perfects;
      for (int n = 2; n <= max_n; ++n) {
        if (is_perfect(n)) {
          perfects.push_back(n);
        }
      }
      std::cout << "Perfect numbers up to " << max_n << ":";
      for (int p : perfects) {
        std::cout << " " << p;
      }
      std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  if (rank == 0) {
    int next_candidate = 2;
    int finished_workers = 0;
    std::vector<int> perfects;

    while (finished_workers < size - 1) {
      int msg = 0;
      MPI_Status status;
      MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

      if (msg > 0) {
        perfects.push_back(msg);
      }

      int response = 0;
      if (next_candidate <= max_n) {
        response = next_candidate;
        ++next_candidate;
      } else {
        response = 0;
        ++finished_workers;
      }

      MPI_Send(&response, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
    }

    std::cout << "Perfect numbers up to " << max_n << ":";
    for (int p : perfects) {
      std::cout << " " << p;
    }
    std::cout << std::endl;
  } else {
    int request = 0;
    MPI_Send(&request, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    while (true) {
      int candidate = 0;
      MPI_Recv(&candidate, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      if (candidate == 0) {
        break;
      }

      int result = is_perfect(candidate) ? candidate : -candidate;
      MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}