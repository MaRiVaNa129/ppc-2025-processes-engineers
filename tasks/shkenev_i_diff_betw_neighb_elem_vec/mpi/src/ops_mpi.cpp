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

void ShkenevIDiffBetwNeighbElemVecMPI::SendToAllProcesses(const std::vector<int> &vec, int my_size, int world_size,
                                                          int base_size, int rem) {
  int current_pos = my_size;
  for (int p = 1; p < world_size; p++) {
    int p_size = base_size + (p < rem ? 1 : 0);
    if (p_size > 0) {
      MPI_Send(vec.data() + current_pos, p_size, MPI_INT, p, 0, MPI_COMM_WORLD);
      current_pos += p_size;
    }
  }
}

void ShkenevIDiffBetwNeighbElemVecMPI::ReceiveFromRoot(std::vector<int> &local_vec, int my_size) {
  if (my_size > 0) {
    MPI_Recv(local_vec.data(), my_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void ShkenevIDiffBetwNeighbElemVecMPI::DistributeData(const std::vector<int> &vec, std::vector<int> &local_vec,
                                                      int world_rank, int my_size, int world_size, int base_size,
                                                      int rem) {
  if (world_rank == 0) {
    std::copy(vec.begin(), vec.begin() + my_size, local_vec.begin());
    SendToAllProcesses(vec, my_size, world_size, base_size, rem);
  } else {
    ReceiveFromRoot(local_vec, my_size);
  }
}

bool ShkenevIDiffBetwNeighbElemVecMPI::RunImpl() {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const std::vector<int> &vec = GetInput();
  int n = static_cast<int>(vec.size());

  bool small_vector = (n < 2);
  if (small_vector) {
    GetOutput() = 0;
    return true;
  }

  int base_size = n / world_size;
  int rem = n % world_size;

  int my_size = base_size;
  int gets_extra = (world_rank < rem);
  my_size = my_size + gets_extra;

  int my_start = world_rank * base_size;
  my_start = my_start + (world_rank < rem ? world_rank : rem);

  std::vector<int> local_vec(my_size);
  DistributeData(vec, local_vec, world_rank, my_size, world_size, base_size, rem);

  int local_max = 0;
  int local_n = static_cast<int>(local_vec.size());

  int i = 0;
  while (i < local_n - 1) {
    int diff = std::abs(local_vec[i + 1] - local_vec[i]);
    local_max = (diff > local_max) ? diff : local_max;
    i++;
  }

  int boundary_diff = 0;
  int has_data = (local_n > 0);
  int not_first_process = (world_rank > 0);
  int not_last_process = (world_rank < world_size - 1);

  int receive_from_left = has_data && not_first_process;
  int send_to_right = has_data && not_last_process;

  int prev_last = 0;
  int my_last = has_data ? local_vec[local_n - 1] : 0;

  if (receive_from_left) {
    MPI_Recv(&prev_last, 1, MPI_INT, world_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    boundary_diff = std::abs(local_vec[0] - prev_last);
  }

  if (send_to_right) {
    MPI_Send(&my_last, 1, MPI_INT, world_rank + 1, 1, MPI_COMM_WORLD);
  }

  local_max = (boundary_diff > local_max) ? boundary_diff : local_max;

  int global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  GetOutput() = global_max;
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
