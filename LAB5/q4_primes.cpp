#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static bool is_prime(int n) {
  if (n < 2) {
    return false;
  }
  if (n == 2) {
    return true;
  }
  if (n % 2 == 0) {
    return false;
  }
  const int limit = static_cast<int>(std::sqrt(static_cast<double>(n)));
  for (int d = 3; d <= limit; d += 2) {
    if (n % d == 0) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int max_n = 100;
  if (argc > 1) {
    max_n = std::atoi(argv[1]);
  }

  if (size < 2) {
    if (rank == 0) {
      std::vector<int> primes;
      for (int n = 2; n <= max_n; ++n) {
        if (is_prime(n)) {
          primes.push_back(n);
        }
      }
      std::cout << "Primes up to " << max_n << ":";
      for (int p : primes) {
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
    std::vector<int> primes;

    while (finished_workers < size - 1) {
      int msg = 0;
      MPI_Status status;
      MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

      if (msg > 0) {
        primes.push_back(msg);
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

    std::cout << "Primes up to " << max_n << ":";
    for (int p : primes) {
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

      int result = is_prime(candidate) ? candidate : -candidate;
      MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}