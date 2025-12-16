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

  if (matrix.empty() || vector.empty()) {
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
    local_cols += 1;
  }

  col_offset = 0;
  for (int i = 0; i < rank; ++i) {
    int cols = base;
    if (i < remainder) {
      cols += 1;
    }
    col_offset += cols;
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
      sum += local_matrix[row * local_cols + col] * local_vector[col];
    }
    local_result[row] = sum;
  }
}

}  // namespace

bool ShkenevImatvectUsingVerticalRibbonMPI::RunImpl() {
  int world_size = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int rows = 0, cols = 0;

  if (rank == 0) {
    rows = static_cast<int>(GetInput().first.size());
    cols = static_cast<int>(GetInput().first[0].size());
  }

  BroadcastMatrixSize(rows, cols);

  if (rows == 0 || cols == 0) {
    return false;
  }

  if (cols < world_size) {
    std::vector<double> result(rows, 0.0);

    if (rank == 0) {
      const auto &matrix = GetInput().first;
      const auto &vector = GetInput().second;

      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          result[i] += matrix[i][j] * vector[j];
        }
      }

      GetOutput() = result;
    } else {
      GetOutput().resize(static_cast<std::size_t>(rows), 0.0);
    }

    double *output_ptr = rows > 0 ? GetOutput().data() : nullptr;
    MPI_Bcast(output_ptr, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return true;
  }

  int local_cols = 0, col_offset = 0;
  ComputeColumnDistribution(rank, world_size, cols, local_cols, col_offset);

  std::vector<double> local_matrix(static_cast<std::size_t>(rows) * local_cols, 0.0);
  std::vector<double> local_vector(local_cols, 0.0);
  std::vector<double> local_result(rows, 0.0);

  if (rank == 0) {
    ScatterMatrixColumns(GetInput().first, local_matrix, rows, local_cols, col_offset);
    ScatterVectorPart(GetInput().second, local_vector, local_cols, col_offset);

    for (int proc = 1; proc < world_size; ++proc) {
      int proc_cols = 0, proc_offset = 0;
      ComputeColumnDistribution(proc, world_size, cols, proc_cols, proc_offset);

      if (proc_cols == 0) {
        continue;
      }

      std::vector<double> temp_matrix(rows * proc_cols, 0.0);
      std::vector<double> temp_vector(proc_cols, 0.0);

      ScatterMatrixColumns(GetInput().first, temp_matrix, rows, proc_cols, proc_offset);
      ScatterVectorPart(GetInput().second, temp_vector, proc_cols, proc_offset);

      MPI_Send(temp_matrix.data(), rows * proc_cols, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
      MPI_Send(temp_vector.data(), proc_cols, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
    }
  } else {
    if (local_cols > 0) {
      MPI_Status status;
      MPI_Recv(local_matrix.data(), rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(local_vector.data(), local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    }
  }

  ComputeLocalProduct(local_matrix, local_vector, local_result, rows, local_cols);

  std::vector<double> result(static_cast<std::size_t>(rows), 0.0);
  double *result_ptr = rows > 0 ? result.data() : nullptr;
  MPI_Reduce(local_result.data(), rank == 0 ? result_ptr : nullptr, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(result_ptr, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = result;
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
