#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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

  const auto &in = GetInput();
  const auto &matrix_a = in.first;
  const auto &matrix_b = in.second;

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  const std::size_t cols_a = matrix_a[0].size();
  for (const auto &row : matrix_a) {
    if (row.size() != cols_a) {
      return false;
    }
  }

  const std::size_t rows_b = matrix_b.size();
  if (rows_b != cols_a) {
    return false;
  }

  const std::size_t cols_b = matrix_b[0].size();
  return std::ranges::all_of(matrix_b, [cols_b](const auto &row) { return row.size() == cols_b; });
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PreProcessingImpl() {
  return true;
}

namespace {

void FillAFlatFromMatrixA(const InType &in, std::vector<double> &a_flat, int rows_a, int cols_a) {
  const auto &matrix_a = in.first;
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_a; ++j) {
      a_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_a) + static_cast<std::size_t>(j)] =
          matrix_a[i][j];
    }
  }
}

void FillBFlatFromMatrixB(const InType &in, std::vector<double> &b_flat, int cols_a, int cols_b) {
  const auto &matrix_b = in.second;
  for (int i = 0; i < cols_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      b_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b) + static_cast<std::size_t>(j)] =
          matrix_b[i][j];
    }
  }
}

void CalculateElementCFromFlattened(const std::vector<double> &a_flat, const std::vector<double> &b_flat,
                                    std::vector<double> &c_flat, int cols_a, int cols_b, int i, int j, int k) {
  const double a_val =
      a_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_a) + static_cast<std::size_t>(j)];
  if (a_val == 0.0) {
    return;
  }

  const double b_val =
      b_flat[static_cast<std::size_t>(j) * static_cast<std::size_t>(cols_b) + static_cast<std::size_t>(k)];
  c_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b) + static_cast<std::size_t>(k)] += a_val * b_val;
}

void CalculateRowCFromFlattened(const std::vector<double> &a_flat, const std::vector<double> &b_flat,
                                std::vector<double> &c_flat, int cols_a, int cols_b, int i) {  // rows_a убран
  for (int j = 0; j < cols_a; ++j) {
    for (int k = 0; k < cols_b; ++k) {
      CalculateElementCFromFlattened(a_flat, b_flat, c_flat, cols_a, cols_b, i, j, k);
    }
  }
}

void CalculateCFlatFromFlattened(const std::vector<double> &a_flat, const std::vector<double> &b_flat,
                                 std::vector<double> &c_flat, int rows_a, int cols_a, int cols_b) {
  for (int i = 0; i < rows_a; ++i) {
    CalculateRowCFromFlattened(a_flat, b_flat, c_flat, cols_a, cols_b, i);  // rows_a убран
  }
}

void FillResultFromCFlat(const std::vector<double> &c_flat, OutType &result, int rows_a, int cols_b) {
  for (int i = 0; i < rows_a; ++i) {
    for (int k = 0; k < cols_b; ++k) {
      result[i][k] =
          c_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b) + static_cast<std::size_t>(k)];
    }
  }
}

void FillSendBufferFromMatrixB(const InType &in, std::vector<double> &send_buffer, int cols_a, int proc_start,
                               int proc_width) {
  const auto &matrix_b = in.second;
  for (int row = 0; row < cols_a; ++row) {
    for (int kk = 0; kk < proc_width; ++kk) {
      const int global_k = proc_start + kk;
      send_buffer[static_cast<std::size_t>(row) * static_cast<std::size_t>(proc_width) + static_cast<std::size_t>(kk)] =
          matrix_b[row][global_k];
    }
  }
}

void DistributeBStripToProcess(const InType &in, int cols_a, int cols_b, int world_size, std::vector<double> &b_strip,
                               int rank) {
  if (rank != 0) {
    return;
  }

  const int base = cols_b / world_size;
  const int remainder = cols_b % world_size;

  for (int proc_idx = 0; proc_idx < world_size; ++proc_idx) {
    const int proc_start = proc_idx * base + std::min(proc_idx, remainder);
    const int proc_width = base + (proc_idx < remainder ? 1 : 0);

    if (proc_width <= 0) {
      continue;
    }

    std::vector<double> send_buffer(static_cast<std::size_t>(cols_a) * static_cast<std::size_t>(proc_width));

    FillSendBufferFromMatrixB(in, send_buffer, cols_a, proc_start, proc_width);

    if (proc_idx == 0) {
      b_strip = std::move(send_buffer);
    } else {
      MPI_Send(send_buffer.data(), cols_a * proc_width, MPI_DOUBLE, proc_idx, 101, MPI_COMM_WORLD);
    }
  }
}

void CalculateElementCStrip(const std::vector<double> &a_flat, const std::vector<double> &b_strip,
                            std::vector<double> &c_strip, int cols_a, int my_width, int i, int j, int k) {
  const double aij =
      a_flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_a) + static_cast<std::size_t>(j)];
  if (aij == 0.0) {
    return;
  }

  c_strip[static_cast<std::size_t>(i) * static_cast<std::size_t>(my_width) + static_cast<std::size_t>(k)] +=
      aij * b_strip[static_cast<std::size_t>(j) * static_cast<std::size_t>(my_width) + static_cast<std::size_t>(k)];
}

void CalculateRowCStrip(const std::vector<double> &a_flat, const std::vector<double> &b_strip,
                        std::vector<double> &c_strip, int cols_a, int my_width, int i) {
  for (int j = 0; j < cols_a; ++j) {
    for (int k = 0; k < my_width; ++k) {
      CalculateElementCStrip(a_flat, b_strip, c_strip, cols_a, my_width, i, j, k);
    }
  }
}

void CalculateCStripFromStrips(const std::vector<double> &a_flat, const std::vector<double> &b_strip,
                               std::vector<double> &c_strip, int rows_a, int cols_a, int my_width) {
  for (int i = 0; i < rows_a; ++i) {
    CalculateRowCStrip(a_flat, b_strip, c_strip, cols_a, my_width, i);
  }
}

void FillFullResultFromCStrip(std::vector<double> &full_result, const std::vector<double> &c_strip, int rows_a,
                              int cols_b, int proc_start, int proc_width) {
  for (int i = 0; i < rows_a; ++i) {
    for (int k = 0; k < proc_width; ++k) {
      const int global_k = proc_start + k;
      full_result[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b) + static_cast<std::size_t>(global_k)] =
          c_strip[static_cast<std::size_t>(i) * static_cast<std::size_t>(proc_width) + static_cast<std::size_t>(k)];
    }
  }
}

void ProcessEmptyResult(int proc_idx) {
  MPI_Status status;
  MPI_Probe(proc_idx, 102, MPI_COMM_WORLD, &status);
  int count = 0;
  MPI_Get_count(&status, MPI_DOUBLE, &count);
  if (count > 0) {
    std::vector<double> dummy(count);
    MPI_Recv(dummy.data(), count, MPI_DOUBLE, proc_idx, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(nullptr, 0, MPI_DOUBLE, proc_idx, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void ProcessNonEmptyResult(int proc_idx, int proc_start, int proc_width, int rows_a, int cols_b,
                           std::vector<double> &full_result) {
  std::vector<double> receive_buffer(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(proc_width));
  MPI_Recv(receive_buffer.data(), rows_a * proc_width, MPI_DOUBLE, proc_idx, 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  FillFullResultFromCStrip(full_result, receive_buffer, rows_a, cols_b, proc_start, proc_width);
}

void GatherResultsFromWorkers(std::vector<double> &full_result, int rows_a, int cols_b, int world_size,
                              const std::vector<double> &c_strip, int my_start, int my_width) {
  const int base = cols_b / world_size;
  const int remainder = cols_b % world_size;

  if (my_width > 0) {
    FillFullResultFromCStrip(full_result, c_strip, rows_a, cols_b, my_start, my_width);
  }

  for (int proc_idx = 1; proc_idx < world_size; ++proc_idx) {
    const int proc_start = proc_idx * base + std::min(proc_idx, remainder);
    const int proc_width = base + (proc_idx < remainder ? 1 : 0);

    if (proc_width <= 0) {
      ProcessEmptyResult(proc_idx);
      continue;
    }

    ProcessNonEmptyResult(proc_idx, proc_start, proc_width, rows_a, cols_b, full_result);
  }
}

void FillResultFromFullResult(const std::vector<double> &full_result, OutType &result, int rows_a, int cols_b) {
  for (int i = 0; i < rows_a; ++i) {
    for (int k = 0; k < cols_b; ++k) {
      result[i][k] =
          full_result[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_b) + static_cast<std::size_t>(k)];
    }
  }
}

void SendResultToWorkers(const std::vector<double> &full_result, int rows_a, int cols_b, int world_size) {
  for (int proc_idx = 1; proc_idx < world_size; ++proc_idx) {
    MPI_Send(full_result.data(), rows_a * cols_b, MPI_DOUBLE, proc_idx, 103, MPI_COMM_WORLD);
  }
}

void ReceiveResultFromMaster(std::vector<double> &full_result, int rows_a, int cols_b) {
  MPI_Recv(full_result.data(), rows_a * cols_b, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    const auto &matrix_a = in.first;
    const auto &matrix_b = in.second;
    rows_a = static_cast<int>(matrix_a.size());
    cols_a = static_cast<int>(matrix_a.empty() ? 0 : matrix_a[0].size());
    cols_b = static_cast<int>(matrix_b.empty() ? 0 : matrix_b[0].size());
  }

  MPI_Bcast(&rows_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_b, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows_a <= 0 || cols_a <= 0 || cols_b < 0) {
    GetOutput() = OutType{};
    return false;
  }

  if (cols_b < world_size) {
    std::vector<double> a_flat(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_a), 0.0);
    std::vector<double> b_flat(static_cast<std::size_t>(cols_a) * static_cast<std::size_t>(cols_b), 0.0);

    if (rank == 0) {
      FillAFlatFromMatrixA(GetInput(), a_flat, rows_a, cols_a);
      FillBFlatFromMatrixB(GetInput(), b_flat, cols_a, cols_b);
    }

    MPI_Bcast(a_flat.data(), rows_a * cols_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_flat.data(), cols_a * cols_b, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> c_flat(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_b), 0.0);
    CalculateCFlatFromFlattened(a_flat, b_flat, c_flat, rows_a, cols_a, cols_b);

    if (rank == 0) {
      OutType result(rows_a, std::vector<double>(cols_b, 0.0));
      FillResultFromCFlat(c_flat, result, rows_a, cols_b);
      GetOutput() = std::move(result);
    } else {
      GetOutput() = OutType(rows_a, std::vector<double>(cols_b, 0.0));
    }

    return true;
  }

  const int base = cols_b / world_size;
  const int remainder = cols_b % world_size;
  const int my_start = rank * base + std::min(rank, remainder);
  const int my_width = base + (rank < remainder ? 1 : 0);

  std::vector<double> a_flat(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_a), 0.0);

  if (rank == 0) {
    FillAFlatFromMatrixA(GetInput(), a_flat, rows_a, cols_a);
  }

  MPI_Bcast(a_flat.data(), rows_a * cols_a, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> b_strip;
  if (my_width > 0) {
    b_strip.assign(static_cast<std::size_t>(cols_a) * static_cast<std::size_t>(my_width), 0.0);
  }

  if (rank == 0) {
    DistributeBStripToProcess(GetInput(), cols_a, cols_b, world_size, b_strip, rank);
  } else if (my_width > 0) {
    MPI_Recv(b_strip.data(), cols_a * my_width, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  std::vector<double> c_strip;
  if (my_width > 0) {
    c_strip.assign(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(my_width), 0.0);
    CalculateCStripFromStrips(a_flat, b_strip, c_strip, rows_a, cols_a, my_width);
  }

  std::vector<double> full_result;
  if (rank == 0) {
    full_result.assign(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_b), 0.0);
    GatherResultsFromWorkers(full_result, rows_a, cols_b, world_size, c_strip, my_start, my_width);
  } else if (my_width > 0) {
    MPI_Send(c_strip.data(), rows_a * my_width, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
  } else {
    MPI_Send(nullptr, 0, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    OutType result(rows_a, std::vector<double>(cols_b, 0.0));
    FillResultFromFullResult(full_result, result, rows_a, cols_b);
    GetOutput() = std::move(result);
    SendResultToWorkers(full_result, rows_a, cols_b, world_size);
  } else {
    full_result.assign(static_cast<std::size_t>(rows_a) * static_cast<std::size_t>(cols_b), 0.0);
    ReceiveResultFromMaster(full_result, rows_a, cols_b);

    OutType result(rows_a, std::vector<double>(cols_b, 0.0));
    FillResultFromFullResult(full_result, result, rows_a, cols_b);
    GetOutput() = std::move(result);
  }

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
