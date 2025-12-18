#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

ShkenevImatvectUsingVerticalRibbonMPI::ShkenevImatvectUsingVerticalRibbonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp = in;
  GetInput().swap(temp);
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

  return vector.size() == cols;
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
    int cols_for_proc = base;
    if (i < remainder) {
      cols_for_proc += 1;
    }
    col_offset += cols_for_proc;
  }
}

void ScatterMatrixColumns(const std::vector<std::vector<double>> &matrix, std::vector<double> &local_matrix, int rows,
                          int local_cols, int col_offset) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < local_cols; ++col) {
      int global_col = col_offset + col;
      std::size_t index =
          (static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols)) + static_cast<std::size_t>(col);
      local_matrix[index] = matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(global_col)];
    }
  }
}

void ScatterVectorPart(const std::vector<double> &vector, std::vector<double> &local_vector, int local_cols,
                       int col_offset) {
  for (int i = 0; i < local_cols; ++i) {
    local_vector[static_cast<std::size_t>(i)] = vector[col_offset + i];
  }
}

void ComputeLocalProduct(const std::vector<double> &local_matrix, const std::vector<double> &local_vector,
                         std::vector<double> &local_result, int rows, int local_cols) {
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < local_cols; ++col) {
      std::size_t index =
          (static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols)) + static_cast<std::size_t>(col);
      sum += local_matrix[index] * local_vector[static_cast<std::size_t>(col)];
    }
    local_result[static_cast<std::size_t>(row)] = sum;
  }
}

void ReceiveDataFromProcess0(int rows, int local_cols, std::vector<double> &local_matrix,
                             std::vector<double> &local_vector) {
  if (local_cols > 0) {
    MPI_Status status;
    MPI_Recv(local_matrix.data(), rows * local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(local_vector.data(), local_cols, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
  }
}

void GatherAndBroadcastResults(int rank, int rows, const std::vector<double> &local_result,
                               std::vector<double> &result) {
  if (rows > 0) {
    double *result_ptr = result.data();
    MPI_Reduce(local_result.data(), (rank == 0) ? result_ptr : nullptr, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(result_ptr, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

}  // namespace

bool ShkenevImatvectUsingVerticalRibbonMPI::HandleSmallMatrixCase(int rank, int rows, int cols) {
  std::vector<double> result(static_cast<std::size_t>(rows), 0.0);

  if (rank == 0) {
    const auto &matrix = GetInput().first;
    const auto &vector = GetInput().second;

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        result[static_cast<std::size_t>(i)] += matrix[i][j] * vector[j];
      }
    }

    GetOutput() = result;
  } else {
    GetOutput().resize(static_cast<std::size_t>(rows), 0.0);
  }

  if (rows > 0) {
    MPI_Bcast(GetOutput().data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  return true;
}

void ShkenevImatvectUsingVerticalRibbonMPI::SendDataToProcesses(int world_size, int rows, int cols) {
  for (int proc = 1; proc < world_size; ++proc) {
    int proc_cols = 0;
    int proc_offset = 0;
    ComputeColumnDistribution(proc, world_size, cols, proc_cols, proc_offset);

    if (proc_cols == 0) {
      continue;
    }

    std::vector<double> temp_matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(proc_cols), 0.0);
    std::vector<double> temp_vector(static_cast<std::size_t>(proc_cols), 0.0);

    ScatterMatrixColumns(GetInput().first, temp_matrix, rows, proc_cols, proc_offset);
    ScatterVectorPart(GetInput().second, temp_vector, proc_cols, proc_offset);

    MPI_Send(temp_matrix.data(), rows * proc_cols, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
    MPI_Send(temp_vector.data(), proc_cols, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
  }
}

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

  if ((rows == 0) || (cols == 0)) {
    return false;
  }

  if (cols < world_size) {
    return HandleSmallMatrixCase(rank, rows, cols);
  }

  int local_cols = 0;
  int col_offset = 0;
  ComputeColumnDistribution(rank, world_size, cols, local_cols, col_offset);

  std::vector<double> local_matrix(static_cast<std::size_t>(rows) * static_cast<std::size_t>(local_cols), 0.0);
  std::vector<double> local_vector(static_cast<std::size_t>(local_cols), 0.0);
  std::vector<double> local_result(static_cast<std::size_t>(rows), 0.0);

  if (rank == 0) {
    ScatterMatrixColumns(GetInput().first, local_matrix, rows, local_cols, col_offset);
    ScatterVectorPart(GetInput().second, local_vector, local_cols, col_offset);
    SendDataToProcesses(world_size, rows, cols);
  } else {
    ReceiveDataFromProcess0(rows, local_cols, local_matrix, local_vector);
  }

  ComputeLocalProduct(local_matrix, local_vector, local_result, rows, local_cols);

  std::vector<double> result(static_cast<std::size_t>(rows), 0.0);
  GatherAndBroadcastResults(rank, rows, local_result, result);

  GetOutput() = result;
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
