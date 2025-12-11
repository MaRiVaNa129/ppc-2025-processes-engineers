#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

ShkenevImatvectUsingVerticalRibbonMPI::ShkenevImatvectUsingVerticalRibbonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ShkenevImatvectUsingVerticalRibbonMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank != 0) {
    return true;
  }

  const auto &in = GetInput();
  const auto &A = in.first;
  const auto &B = in.second;

  if (A.empty() || B.empty()) {
    return false;
  }
  std::size_t colsA = A[0].size();
  for (const auto &row : A) {
    if (row.size() != colsA) {
      return false;
    }
  }
  std::size_t brow = B.size();
  if (brow != colsA) {
    return false;
  }
  std::size_t colsB = B[0].size();
  for (const auto &row : B) {
    if (row.size() != colsB) {
      return false;
    }
  }

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PreProcessingImpl() {
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::RunImpl() {
  auto internal_start = std::chrono::high_resolution_clock::now();

  int world_size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rowsA = 0, colsA = 0, colsB = 0;
  if (rank == 0) {
    const auto &in = GetInput();
    const auto &A = in.first;
    const auto &B = in.second;
    rowsA = static_cast<int>(A.size());
    colsA = static_cast<int>(A.empty() ? 0 : A[0].size());
    colsB = static_cast<int>(B.empty() ? 0 : B[0].size());
  }

  MPI_Bcast(&rowsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colsA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colsB, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rowsA <= 0 || colsA <= 0 || colsB < 0) {
    GetOutput() = OutType{};
    return false;
  }

  if (colsB < world_size) {
    std::vector<double> Aflat(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(colsA), 0.0);
    std::vector<double> Bflat(static_cast<std::size_t>(colsA) * static_cast<std::size_t>(colsB), 0.0);

    if (rank == 0) {
      const auto &in = GetInput();
      const auto &A = in.first;
      const auto &B = in.second;

      for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
          Aflat[static_cast<std::size_t>(i) * colsA + j] = A[i][j];
        }
      }

      for (int i = 0; i < colsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
          Bflat[static_cast<std::size_t>(i) * colsB + j] = B[i][j];
        }
      }
    }

    MPI_Bcast(Aflat.data(), rowsA * colsA, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Bflat.data(), colsA * colsB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> Cflat(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(colsB), 0.0);

    for (int i = 0; i < rowsA; ++i) {
      for (int j = 0; j < colsA; ++j) {
        double a_val = Aflat[static_cast<std::size_t>(i) * colsA + j];
        if (a_val == 0.0) {
          continue;
        }

        for (int k = 0; k < colsB; ++k) {
          double b_val = Bflat[static_cast<std::size_t>(j) * colsB + k];
          Cflat[static_cast<std::size_t>(i) * colsB + k] += a_val * b_val;
        }
      }
    }

    OutType result(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i) {
      for (int k = 0; k < colsB; ++k) {
        result[i][k] = Cflat[static_cast<std::size_t>(i) * colsB + k];
      }
    }

    GetOutput() = std::move(result);
    return true;
  }

  int base = colsB / world_size;
  int rem = colsB % world_size;
  int my_start = rank * base + std::min(rank, rem);
  int my_width = base + (rank < rem ? 1 : 0);

  std::vector<double> Aflat(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(colsA), 0.0);
  std::vector<double> Bstrip;
  if (my_width > 0) {
    Bstrip.assign(static_cast<std::size_t>(colsA) * static_cast<std::size_t>(my_width), 0.0);
  }

  const int TAG_B = 101;
  const int TAG_C = 102;

  if (rank == 0) {
    const auto &in = GetInput();
    const auto &A = in.first;
    const auto &B = in.second;

    for (int i = 0; i < rowsA; ++i) {
      for (int j = 0; j < colsA; ++j) {
        Aflat[static_cast<std::size_t>(i) * colsA + j] = A[i][j];
      }
    }

    MPI_Bcast(Aflat.data(), rowsA * colsA, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int p = 0; p < world_size; ++p) {
      int p_start = p * base + std::min(p, rem);
      int p_width = base + (p < rem ? 1 : 0);
      if (p_width <= 0) {
        continue;
      }

      std::vector<double> sendbuf(static_cast<std::size_t>(colsA) * static_cast<std::size_t>(p_width));
      for (int row = 0; row < colsA; ++row) {
        for (int kk = 0; kk < p_width; ++kk) {
          int global_k = p_start + kk;
          sendbuf[static_cast<std::size_t>(row) * p_width + kk] = B[row][global_k];
        }
      }

      if (p == 0) {
        Bstrip = std::move(sendbuf);
      } else {
        MPI_Send(sendbuf.data(), colsA * p_width, MPI_DOUBLE, p, TAG_B, MPI_COMM_WORLD);
      }
    }
  } else {
    MPI_Bcast(Aflat.data(), rowsA * colsA, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_width > 0) {
      MPI_Recv(Bstrip.data(), colsA * my_width, MPI_DOUBLE, 0, TAG_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  std::vector<double> Cstrip;
  if (my_width > 0) {
    Cstrip.assign(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(my_width), 0.0);

    for (int i = 0; i < rowsA; ++i) {
      for (int j = 0; j < colsA; ++j) {
        double aij = Aflat[static_cast<std::size_t>(i) * colsA + j];
        if (aij == 0.0) {
          continue;
        }
        for (int k = 0; k < my_width; ++k) {
          Cstrip[static_cast<std::size_t>(i) * my_width + k] +=
              aij * Bstrip[static_cast<std::size_t>(j) * my_width + k];
        }
      }
    }
  }

  std::vector<double> full_result;
  if (rank == 0) {
    full_result.assign(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(colsB), 0.0);

    if (my_width > 0) {
      for (int i = 0; i < rowsA; ++i) {
        for (int k = 0; k < my_width; ++k) {
          int global_k = my_start + k;
          full_result[static_cast<std::size_t>(i) * colsB + global_k] =
              Cstrip[static_cast<std::size_t>(i) * my_width + k];
        }
      }
    }

    for (int p = 1; p < world_size; ++p) {
      int p_start = p * base + std::min(p, rem);
      int p_width = base + (p < rem ? 1 : 0);

      if (p_width <= 0) {
        MPI_Status status;
        MPI_Probe(p, TAG_C, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        if (count > 0) {
          std::vector<double> dummy(count);
          MPI_Recv(dummy.data(), count, MPI_DOUBLE, p, TAG_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(nullptr, 0, MPI_DOUBLE, p, TAG_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        continue;
      }

      std::vector<double> recvbuf(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(p_width));
      MPI_Recv(recvbuf.data(), rowsA * p_width, MPI_DOUBLE, p, TAG_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < rowsA; ++i) {
        for (int k = 0; k < p_width; ++k) {
          int global_k = p_start + k;
          full_result[static_cast<std::size_t>(i) * colsB + global_k] =
              recvbuf[static_cast<std::size_t>(i) * p_width + k];
        }
      }
    }
  } else {
    if (my_width > 0) {
      MPI_Send(Cstrip.data(), rowsA * my_width, MPI_DOUBLE, 0, TAG_C, MPI_COMM_WORLD);
    } else {
      MPI_Send(nullptr, 0, MPI_DOUBLE, 0, TAG_C, MPI_COMM_WORLD);
    }
  }

  if (rank == 0) {
    OutType result(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i) {
      for (int k = 0; k < colsB; ++k) {
        result[i][k] = full_result[static_cast<std::size_t>(i) * colsB + k];
      }
    }
    GetOutput() = std::move(result);

    for (int p = 1; p < world_size; ++p) {
      MPI_Send(full_result.data(), rowsA * colsB, MPI_DOUBLE, p, 103, MPI_COMM_WORLD);
    }
  } else {
    full_result.assign(static_cast<std::size_t>(rowsA) * static_cast<std::size_t>(colsB), 0.0);
    MPI_Recv(full_result.data(), rowsA * colsB, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    OutType result(rowsA, std::vector<double>(colsB, 0.0));
    for (int i = 0; i < rowsA; ++i) {
      for (int k = 0; k < colsB; ++k) {
        result[i][k] = full_result[static_cast<std::size_t>(i) * colsB + k];
      }
    }
    GetOutput() = std::move(result);
  }

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
