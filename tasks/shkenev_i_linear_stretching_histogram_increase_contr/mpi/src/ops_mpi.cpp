#include "shkenev_i_linear_stretching_histogram_increase_contr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

ShkenevIlinerStretchingHistIncreaseContrMPI::ShkenevIlinerStretchingHistIncreaseContrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIlinerStretchingHistIncreaseContrMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 1;

  if (rank == 0) {
    if (GetInput().empty()) {
      is_valid = 0;
    } else {
      for (int val : GetInput()) {
        if (val < 0 || val > 255) {
          is_valid = 0;
          break;
        }
      }
    }
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool ShkenevIlinerStretchingHistIncreaseContrMPI::PreProcessingImpl() {
  return true;
}

bool ShkenevIlinerStretchingHistIncreaseContrMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_size = (rank == 0) ? static_cast<int>(GetInput().size()) : 0;
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_size == 0) {
    if (rank == 0) {
      GetOutput().clear();
    }
    return true;
  }

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  int base_chunk = total_size / size;
  int remainder = total_size % size;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = base_chunk + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  std::vector<int> local_data(send_counts[rank]);

  if (rank == 0) {
    MPI_Scatterv(GetInput().data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[rank],
                 MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Scatterv(nullptr, send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[rank], MPI_INT, 0,
                 MPI_COMM_WORLD);
  }

  int local_min = INT_MAX;
  int local_max = INT_MIN;

  if (!local_data.empty()) {
    auto [mi, ma] = std::minmax_element(local_data.begin(), local_data.end());
    local_min = *mi;
    local_max = *ma;
  }

  int global_min, global_max;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (global_max > global_min) {
    int range = global_max - global_min;
    for (auto &pixel : local_data) {
      pixel = (pixel - global_min) * 255 / range;
    }
  }

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
