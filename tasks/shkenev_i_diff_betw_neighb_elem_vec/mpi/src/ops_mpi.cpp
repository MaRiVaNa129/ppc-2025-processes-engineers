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

int ShkenevIDiffBetwNeighbElemVecMPI::HandleSmallVector(const std::vector<int> &vec, int n) {
  int result = 0;
  for (int i = 0; i < n - 1; i++) {
    result = std::max(result, std::abs(vec[i + 1] - vec[i]));
  }
  return result;
}

void ShkenevIDiffBetwNeighbElemVecMPI::ComputeCountsAndDispls(int n, int world_size, std::vector<int> &cnt,
                                                              std::vector<int> &disp) {
  int base_size = n / world_size;
  int rem = n % world_size;

  int shift = 0;
  for (int i = 0; i < world_size; ++i) {
    if (i < rem) {
      cnt[i] = base_size + 1;
    } else {
      cnt[i] = base_size;
    }
    disp[i] = shift;
    shift += cnt[i];
  }
}

void ShkenevIDiffBetwNeighbElemVecMPI::ScatterData(const std::vector<int> &vec, const std::vector<int> &cnt,
                                                   const std::vector<int> &disp, std::vector<int> &l_vec,
                                                   int world_rank) {
  int l_n = cnt[world_rank];

  if (world_rank == 0) {
    if (l_n > 0) {
      std::copy(vec.begin(), vec.begin() + l_n, l_vec.begin());
    }
    for (int proc = 1; proc < static_cast<int>(cnt.size()); ++proc) {
      if (cnt[proc] > 0) {
        MPI_Send(vec.data() + disp[proc], cnt[proc], MPI_INT, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    if (l_n > 0) {
      MPI_Recv(l_vec.data(), l_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

int ShkenevIDiffBetwNeighbElemVecMPI::LocalCompute(const std::vector<int> &l_vec) {
  int l_max = 0;
  int l_n = static_cast<int>(l_vec.size());
  for (int i = 0; i < l_n - 1; ++i) {
    l_max = std::max(l_max, std::abs(l_vec[i + 1] - l_vec[i]));
  }
  return l_max;
}

int ShkenevIDiffBetwNeighbElemVecMPI::BoundaryExchange(const std::vector<int> &l_vec, int world_rank, int world_size) {
  int l_n = static_cast<int>(l_vec.size());
  int boundary = 0;

  if (world_rank > 0 && l_n > 0) {
    int prev_last = 0;
    MPI_Recv(&prev_last, 1, MPI_INT, world_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    boundary = std::abs(l_vec[0] - prev_last);
  }

  if (world_rank < world_size - 1 && l_n > 0) {
    int my_last = l_vec.back();
    MPI_Send(&my_last, 1, MPI_INT, world_rank + 1, 1, MPI_COMM_WORLD);
  }

  return boundary;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::RunImpl() {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::vector<int> &vec = GetInput();
  int n = static_cast<int>(vec.size());

  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  if (world_size > n) {
    int result = (world_rank == 0 ? HandleSmallVector(vec, n) : 0);
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  std::vector<int> cnt(world_size), disp(world_size);
  ComputeCountsAndDispls(n, world_size, cnt, disp);

  std::vector<int> l_vec(cnt[world_rank]);
  ScatterData(vec, cnt, disp, l_vec, world_rank);

  int local_max = LocalCompute(l_vec);
  int boundary = BoundaryExchange(l_vec, world_rank, world_size);
  local_max = std::max(local_max, boundary);

  int global_max = 0;
  MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_max, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
