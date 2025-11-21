#include "shkenev_i_diff_betw_neighb_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

namespace {

int HandleSmallVector(const std::vector<int> &vec, int n) {
  int result = 0;
  for (int i = 0; i < n - 1; ++i) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    result = std::max(result, diff);
  }
  return result;
}

void ComputeCountsAndDispls(int n, int world_size, std::vector<int> &cnt, std::vector<int> &disp) {
  int base_size = n / world_size;
  int rem = n % world_size;

  int shift = 0;
  for (int i = 0; i < world_size; ++i) {
    cnt[i] = base_size + (i < rem ? 1 : 0);
    disp[i] = shift;
    shift += cnt[i];
  }
}

void ScatterData(const std::vector<int> &vec, const std::vector<int> &cnt, const std::vector<int> &disp,
                 std::vector<int> &l_vec, int world_rank) {
  MPI_Scatterv((world_rank == 0 ? const_cast<int *>(vec.data()) : nullptr), cnt.data(), disp.data(), MPI_INT,
               l_vec.data(), static_cast<int>(l_vec.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

int LocalCompute(const std::vector<int> &l_vec) {
  int l_max = 0;
  int l_n = static_cast<int>(l_vec.size());

  for (int i = 0; i < l_n - 1; ++i) {
    int diff = std::abs(l_vec[i + 1] - l_vec[i]);
    l_max = std::max(l_max, diff);
  }

  return l_max;
}

}  // namespace

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
    int result = (world_rank == 0 ? HandleSmallVector(vec, n) : 0);
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = result;
    return true;
  }

  std::vector<int> cnt(world_size);
  std::vector<int> disp(world_size);
  ComputeCountsAndDispls(n, world_size, cnt, disp);

  std::vector<int> l_vec(cnt[world_rank]);
  ScatterData(vec, cnt, disp, l_vec, world_rank);

  int local_max = LocalCompute(l_vec);

  int last_elem = l_vec.back();
  std::vector<int> all_last(world_size);
  MPI_Allgather(&last_elem, 1, MPI_INT, all_last.data(), 1, MPI_INT, MPI_COMM_WORLD);

  int boundary_max = 0;
  if (world_rank > 0) {
    boundary_max = std::abs(l_vec[0] - all_last[world_rank - 1]);
  }
  local_max = std::max(local_max, boundary_max);

  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
