#include "shkenev_i_linear_stretching_histogram_increase_contr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <utility>
#include <vector>

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

ShkenevIlinerStretchingHistIncreaseContrMPI::ShkenevIlinerStretchingHistIncreaseContrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIlinerStretchingHistIncreaseContrMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 1;

  if (rank == 0) {
    if (!GetInput().empty()) {
      is_valid =
          std::all_of(GetInput().begin(), GetInput().end(), [](int val) { return val >= 0 && val <= 255; }) ? 1 : 0;
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool ShkenevIlinerStretchingHistIncreaseContrMPI::PreProcessingImpl() {
  return true;
}

namespace {
std::pair<std::vector<int>, std::vector<int>> CalculateDistribution(int total_size, int num_processes) {
  std::vector<int> send_counts(num_processes);
  std::vector<int> displs(num_processes);

  const int base_chunk = total_size / num_processes;
  const int remainder = total_size % num_processes;

  for (int i = 0; i < num_processes; ++i) {
    send_counts[i] = base_chunk;
    if (i < remainder) {
      send_counts[i] += 1;
    }

    if (i == 0) {
      displs[i] = 0;
    } else {
      displs[i] = displs[i - 1] + send_counts[i - 1];
    }
  }

  return {send_counts, displs};
}

std::pair<int, int> FindLocalMinMax(const std::vector<int> &data) {
  if (data.empty()) {
    return {INT_MAX, INT_MIN};
  }

  auto min_it = std::ranges::min_element(data);
  auto max_it = std::ranges::max_element(data);
  return {*min_it, *max_it};
}

void ApplyLinearStretching(std::vector<int> &data, int global_min, int global_max) {
  if (global_max <= global_min || data.empty()) {
    return;
  }

  const int range = global_max - global_min;
  for (auto &pixel : data) {
    pixel = (pixel - global_min) * 255 / range;
  }
}

}  // namespace

bool ShkenevIlinerStretchingHistIncreaseContrMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return true;
  }

  auto [send_counts, displs] = CalculateDistribution(total_size, size);

  std::vector<int> local_data(send_counts[rank]);
  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[rank],
                 MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[rank], MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  auto [local_min, local_max] = FindLocalMinMax(local_data);

  int global_min = INT_MAX;
  int global_max = INT_MIN;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  ApplyLinearStretching(local_data, global_min, global_max);

  if (rank == 0) {
    GetOutput().resize(total_size);
  }

  MPI_Gatherv(local_data.data(), send_counts[rank], MPI_INT, (rank == 0) ? GetOutput().data() : nullptr,
              send_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool ShkenevIlinerStretchingHistIncreaseContrMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_linear_stretching_histogram_increase_contr
