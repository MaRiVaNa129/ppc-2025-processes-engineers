#include "shkenev_i_diff_betw_neighb_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

ShkenevIDiffBetwNeighbElemVecMPI::ShkenevIDiffBetwNeighbElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::ValidationImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PreProcessingImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::vector<int> &vec = GetInput();
  int n = static_cast<int>(vec.size());

  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  if (world_size > n) {
    int result = 0;
    if (world_rank == 0) {
      for (int i = 0; i < n - 1; ++i) {
        int diff = std::abs(vec[i + 1] - vec[i]);
        result = std::max(result, diff);
      }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  int base_size = n / world_size;
  int remainder = n % world_size;

  std::vector<int> cnt(world_size);
  std::vector<int> disp(world_size);
  int shift = 0;
  for (int i = 0; i < world_size; ++i) {
    cnt[i] = base_size;
    if (i < remainder) {
      cnt[i] = base_size + 1;
    }
    disp[i] = shift;
    shift += cnt[i];
  }

  int l_n = cnt[world_rank];
  std::vector<int> l_vec(l_n);

  if (world_rank == 0) {
    if (l_n > 0) {
      std::copy(vec.begin(), vec.begin() + l_n, l_vec.begin());
    }

    for (int proc = 1; proc < world_size; ++proc) {
      if (cnt[proc] > 0) {
        MPI_Send(vec.data() + disp[proc], cnt[proc], MPI_INT, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (l_n > 0) {
      MPI_Recv(l_vec.data(), l_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  int l_max = 0;
  for (int i = 0; i < l_n - 1; ++i) {
    int diff = std::abs(l_vec[i + 1] - l_vec[i]);
    l_max = std::max(l_max, diff);
  }

  if (world_size > 1) {
    if (world_rank > 0 && l_n > 0) {
      int prev_last = 0;
      MPI_Recv(&prev_last, 1, MPI_INT, world_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int boundary_diff = std::abs(l_vec[0] - prev_last);
      l_max = std::max(l_max, boundary_diff);
    }

    if (world_rank < world_size - 1 && l_n > 0) {
      int my_last = l_vec[l_n - 1];
      MPI_Send(&my_last, 1, MPI_INT, world_rank + 1, 1, MPI_COMM_WORLD);
    }
  }

  int global_max = 0;
  MPI_Reduce(&l_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;

  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
