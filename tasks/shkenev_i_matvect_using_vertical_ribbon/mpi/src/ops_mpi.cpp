#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
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

  const auto &input = GetInput();
  const auto &matrix = input.first;
  const auto &vector = input.second;

  if (matrix.empty()) {
    return false;
  }

  if (vector.empty()) {
    return false;
  }

  std::size_t cols = matrix[0].size();

  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  if (vector.size() != cols) {
    return false;
  }

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PreProcessingImpl() {
  return true;
}

namespace {

void BroadcastMatrixSize(int &rows, int &cols) {
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void ComputeColumnDistribution(int rank, int world_size, int total_cols, int &local_cols, int &col_offset) {
  int base = total_cols / world_size;
  int remainder = total_cols % world_size;

  local_cols = base;
  if (rank < remainder) {
    local_cols = local_cols + 1;
  }

  col_offset = 0;
  for (int i = 0; i < rank; ++i) {
    int cols = base;
    if (i < remainder) {
      cols = cols + 1;
    }
    col_offset = col_offset + cols;
  }
}

void ScatterMatrixColumns(const std::vector<std::vector<double>> &matrix, std::vector<double> &local_matrix, int rows,
                          int local_cols, int col_offset) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < local_cols; ++col) {
      int global_col = col_offset + col;
      local_matrix[row * local_cols + col] = matrix[row][global_col];
    }
  }
}

void ScatterVectorPart(const std::vector<double> &vector, std::vector<double> &local_vector, int local_cols,
                       int col_offset) {
  for (int i = 0; i < local_cols; ++i) {
    local_vector[i] = vector[col_offset + i];
  }
}

void ComputeLocalProduct(const std::vector<double> &local_matrix, const std::vector<double> &local_vector,
                         std::vector<double> &local_result, int rows, int local_cols) {
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < local_cols; ++col) {
      sum = sum + local_matrix[row * local_cols + col] * local_vector[col];
    }
    local_result[row] = sum;
  }
}

}  // namespace

bool ShkenevImatvectUsingVerticalRibbonMPI::RunImpl() {
  int world_size = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    rows = static_cast<int>(GetInput().first.size());
    cols = static_cast<int>(GetInput().first[0].size());
  }

  BroadcastMatrixSize(rows, cols);

  if (rows == 0 || cols == 0) {
    return false;
  }

  std::vector<int> local_cols_vec(world_size, cols / world_size);
  int remainder = cols % world_size;
  for (int i = 0; i < remainder; ++i) {
    local_cols_vec[i]++;
  }

  std::vector<int> displs(world_size, 0);
  for (int i = 1; i < world_size; ++i) {
    displs[i] = displs[i - 1] + local_cols_vec[i - 1];
  }

  int local_cols = local_cols_vec[rank];
  int col_offset = displs[rank];

  std::vector<double> local_matrix(static_cast<size_t>(rows) * local_cols);

  if (rank == 0) {
    std::vector<double> flat_matrix(rows * cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        flat_matrix[r * cols + c] = GetInput().first[r][c];
      }
    }

    std::vector<int> sendcounts(world_size);
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = rows * local_cols_vec[i];
    }

    std::vector<int> senddispls(world_size);
    senddispls[0] = 0;
    for (int i = 1; i < world_size; ++i) {
      senddispls[i] = senddispls[i - 1] + sendcounts[i - 1];
    }

    MPI_Scatterv(flat_matrix.data(), sendcounts.data(), senddispls.data(), MPI_DOUBLE, local_matrix.data(),
                 rows * local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE, local_matrix.data(), rows * local_cols, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
  }

  std::vector<double> local_vector(local_cols);
  if (rank == 0) {
    for (int i = 0; i < local_cols; ++i) {
      local_vector[i] = GetInput().second[col_offset + i];
    }
  }
  MPI_Bcast(local_vector.data(), local_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_result(rows, 0.0);
  ComputeLocalProduct(local_matrix, local_vector, local_result, rows, local_cols);

  std::vector<double> result(rows, 0.0);
  MPI_Allreduce(local_result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = result;
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
