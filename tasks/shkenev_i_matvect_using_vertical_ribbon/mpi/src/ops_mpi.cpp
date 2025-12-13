#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"

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
  const auto &a = in.first;
  const auto &b = in.second;

  if (a.empty() || b.empty()) {
    return false;
  }

  size_t cols_a = a[0].size();

  if (!std::ranges::all_of(a, [cols_a](const auto &row) { return row.size() == cols_a; })) {
    return false;
  }

  size_t rows_b = b.size();
  if (rows_b != cols_a) {
    return false;
  }

  size_t cols_b = b[0].size();

  if (!std::ranges::all_of(b, [cols_b](const auto &row) { return row.size() == cols_b; })) {
    return false;
  }

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PreProcessingImpl() {
  return true;
}

namespace {
void BroadcastMatrixDimensions(int &rows_a, int &cols_a, int &cols_b) {
  MPI_Bcast(&rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void FlattenMatrices(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b,
                     std::vector<double> &aflat, std::vector<double> &bflat, int rows_a, int cols_a, int cols_b) {
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_a; ++j) {
      aflat[(static_cast<size_t>(i) * cols_a) + j] = a[i][j];
    }
  }

  for (int i = 0; i < cols_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      bflat[(static_cast<size_t>(i) * cols_b) + j] = b[i][j];
    }
  }
}

void SerialMultiplication(const std::vector<double> &aflat, const std::vector<double> &bflat,
                          std::vector<double> &cflat, int rows_a, int cols_a, int cols_b) {
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_a; ++j) {
      double a_val = aflat[(static_cast<size_t>(i) * cols_a) + j];
      if (a_val == 0.0) {
        continue;
      }

      for (int k = 0; k < cols_b; ++k) {
        double b_val = bflat[(static_cast<size_t>(j) * cols_b) + k];
        cflat[(static_cast<size_t>(i) * cols_b) + k] += a_val * b_val;
      }
    }
  }
}

void DistributeStripes(int world_size, const std::vector<std::vector<double>> &b, int base, int rem,
                       std::vector<double> &bstrip, int rank, int cols_a, int my_width) {
  const int tag_b = 101;

  if (rank == 0) {
    for (int proc = 0; proc < world_size; ++proc) {
      int proc_start = proc * base;
      if (proc < rem) {
        proc_start += proc;
      } else {
        proc_start += rem;
      }

      int proc_width = base;
      if (proc < rem) {
        proc_width += 1;
      }

      if (proc_width <= 0) {
        continue;
      }

      std::vector<double> sendbuf(static_cast<size_t>(cols_a) * static_cast<size_t>(proc_width));
      for (int row = 0; row < cols_a; ++row) {
        for (int kk = 0; kk < proc_width; ++kk) {
          int global_k = proc_start + kk;
          sendbuf[(static_cast<size_t>(row) * proc_width) + kk] = b[row][global_k];
        }
      }

      if (proc == 0) {
        bstrip = std::move(sendbuf);
      } else {
        MPI_Send(sendbuf.data(), cols_a * proc_width, MPI_DOUBLE, proc, tag_b, MPI_COMM_WORLD);
      }
    }
  } else if (my_width > 0) {
    MPI_Recv(bstrip.data(), cols_a * my_width, MPI_DOUBLE, 0, tag_b, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void MultiplyStrip(const std::vector<double> &aflat, const std::vector<double> &bstrip, std::vector<double> &cstrip,
                   int rows_a, int cols_a, int my_width) {
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_a; ++j) {
      double aij = aflat[(static_cast<size_t>(i) * cols_a) + j];
      if (aij == 0.0) {
        continue;
      }
      for (int k = 0; k < my_width; ++k) {
        cstrip[(static_cast<size_t>(i) * my_width) + k] += aij * bstrip[(static_cast<size_t>(j) * my_width) + k];
      }
    }
  }
}

void GatherResults(int world_size, int rank, int rows_a, int cols_b, int base, int rem,
                   const std::vector<double> &cstrip, int my_width, int my_start, std::vector<double> &full_result) {
  const int tag_c = 102;

  if (rank == 0) {
    if (my_width > 0) {
      for (int i = 0; i < rows_a; ++i) {
        for (int k = 0; k < my_width; ++k) {
          int global_k = my_start + k;
          full_result[(static_cast<size_t>(i) * cols_b) + global_k] = cstrip[(static_cast<size_t>(i) * my_width) + k];
        }
      }
    }

    for (int proc = 1; proc < world_size; ++proc) {
      int proc_start = proc * base;
      if (proc < rem) {
        proc_start += proc;
      } else {
        proc_start += rem;
      }

      int proc_width = base;
      if (proc < rem) {
        proc_width += 1;
      }

      if (proc_width <= 0) {
        MPI_Status status;
        MPI_Probe(proc, tag_c, MPI_COMM_WORLD, &status);
        int count = 0;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        if (count > 0) {
          std::vector<double> dummy(count);
          MPI_Recv(dummy.data(), count, MPI_DOUBLE, proc, tag_c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(nullptr, 0, MPI_DOUBLE, proc, tag_c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        continue;
      }

      std::vector<double> recvbuf(static_cast<size_t>(rows_a) * static_cast<size_t>(proc_width));
      MPI_Recv(recvbuf.data(), rows_a * proc_width, MPI_DOUBLE, proc, tag_c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < rows_a; ++i) {
        for (int k = 0; k < proc_width; ++k) {
          int global_k = proc_start + k;
          full_result[(static_cast<size_t>(i) * cols_b) + global_k] =
              recvbuf[(static_cast<size_t>(i) * proc_width) + k];
        }
      }
    }
  } else if (my_width > 0) {
    MPI_Send(cstrip.data(), rows_a * my_width, MPI_DOUBLE, 0, tag_c, MPI_COMM_WORLD);
  } else {
    MPI_Send(nullptr, 0, MPI_DOUBLE, 0, tag_c, MPI_COMM_WORLD);
  }
}

void BroadcastFullResult(int rank, int world_size, const std::vector<double> &full_result,
                         std::vector<double> &local_full_result, int rows_a, int cols_b) {
  const int tag_result = 103;

  if (rank == 0) {
    for (int proc = 1; proc < world_size; ++proc) {
      MPI_Send(full_result.data(), rows_a * cols_b, MPI_DOUBLE, proc, tag_result, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_full_result.data(), rows_a * cols_b, MPI_DOUBLE, 0, tag_result, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void CreateOutputMatrix(const std::vector<double> &flat_matrix, OutType &result, int rows_a, int cols_b) {
  for (int i = 0; i < rows_a; ++i) {
    for (int k = 0; k < cols_b; ++k) {
      result[i][k] = flat_matrix[(static_cast<size_t>(i) * cols_b) + k];
    }
  }
}

bool AreDimensionsValid(int rows_a, int cols_a, int cols_b) {
  return rows_a > 0 && cols_a > 0 && cols_b >= 0;
}

}  // namespace

bool ShkenevImatvectUsingVerticalRibbonMPI::RunImpl() {
  int world_size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows_a = 0;
  int cols_a = 0;
  int cols_b = 0;

  if (rank == 0) {
    const auto &in = GetInput();
    const auto &a = in.first;
    const auto &b = in.second;
    rows_a = static_cast<int>(a.size());
    if (a.empty()) {
      cols_a = 0;
    } else {
      cols_a = static_cast<int>(a[0].size());
    }

    if (b.empty()) {
      cols_b = 0;
    } else {
      cols_b = static_cast<int>(b[0].size());
    }
  }

  BroadcastMatrixDimensions(rows_a, cols_a, cols_b);

  if (!AreDimensionsValid(rows_a, cols_a, cols_b)) {
    GetOutput() = OutType{};
    return false;
  }

  if (cols_b < world_size) {
    std::vector<double> aflat(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_a), 0.0);
    std::vector<double> bflat(static_cast<size_t>(cols_a) * static_cast<size_t>(cols_b), 0.0);

    if (rank == 0) {
      const auto &in = GetInput();
      const auto &a = in.first;
      const auto &b = in.second;
      FlattenMatrices(a, b, aflat, bflat, rows_a, cols_a, cols_b);
    }

    MPI_Bcast(aflat.data(), rows_a * cols_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bflat.data(), cols_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> cflat(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b), 0.0);
    SerialMultiplication(aflat, bflat, cflat, rows_a, cols_a, cols_b);

    OutType result(rows_a, std::vector<double>(cols_b, 0.0));
    CreateOutputMatrix(cflat, result, rows_a, cols_b);
    GetOutput() = std::move(result);
    return true;
  }

  int base = cols_b / world_size;
  int rem = cols_b % world_size;
  int my_start = rank * base;
  if (rank < rem) {
    my_start += rank;
  } else {
    my_start += rem;
  }

  int my_width = base;
  if (rank < rem) {
    my_width += 1;
  }

  std::vector<double> aflat(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_a), 0.0);
  std::vector<double> bstrip;
  if (my_width > 0) {
    bstrip.assign(static_cast<size_t>(cols_a) * static_cast<size_t>(my_width), 0.0);
  }

  if (rank == 0) {
    const auto &in = GetInput();
    const auto &a = in.first;

    for (int i = 0; i < rows_a; ++i) {
      for (int j = 0; j < cols_a; ++j) {
        aflat[(static_cast<size_t>(i) * cols_a) + j] = a[i][j];
      }
    }
  }

  MPI_Bcast(aflat.data(), rows_a * cols_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const auto &in = GetInput();
    const auto &b = in.second;
    DistributeStripes(world_size, b, base, rem, bstrip, rank, cols_a, my_width);
  } else if (my_width > 0) {
    MPI_Recv(bstrip.data(), cols_a * my_width, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<double> cstrip;
  if (my_width > 0) {
    cstrip.assign(static_cast<size_t>(rows_a) * static_cast<size_t>(my_width), 0.0);
    MultiplyStrip(aflat, bstrip, cstrip, rows_a, cols_a, my_width);
  }

  std::vector<double> full_result;
  if (rank == 0) {
    full_result.assign(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b), 0.0);
  }

  GatherResults(world_size, rank, rows_a, cols_b, base, rem, cstrip, my_width, my_start, full_result);

  std::vector<double> local_full_result;
  if (rank != 0) {
    local_full_result.assign(static_cast<size_t>(rows_a) * static_cast<size_t>(cols_b), 0.0);
  }

  BroadcastFullResult(rank, world_size, full_result, local_full_result, rows_a, cols_b);

  OutType result(rows_a, std::vector<double>(cols_b, 0.0));
  if (rank == 0) {
    CreateOutputMatrix(full_result, result, rows_a, cols_b);
  } else {
    CreateOutputMatrix(local_full_result, result, rows_a, cols_b);
  }

  GetOutput() = std::move(result);
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
