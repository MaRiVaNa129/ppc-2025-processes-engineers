#include "shkenev_i_diff_betw_neighb_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <random>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

ShkenevIDiffBetwNeighbElemVecMPI::ShkenevIDiffBetwNeighbElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::ValidationImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PreProcessingImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::RunImpl() {
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> &vec = GetInput();
  int n = vec.size();
  if (n < 2) {
    if (world_rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  if (world_size > n) {
    if (world_rank == 0) {
      int max_diff = 0;
      for (int i = 0; i < n - 1; i++) {
        int diff = std::abs(vec[i + 1] - vec[i]);
        if (diff > max_diff) {
          max_diff = diff;
        }
      }
      GetOutput() = max_diff;
    }
    return true;
  }

  int min_proc = n / world_size;
  int minus_proc = n % world_size;

  int l_n = min_proc;
  if (world_rank < minus_proc) {
    l_n += 1;
  }

  std::vector<int> l_vec(l_n);

  if (world_rank == 0) {
    int flug = 0;
    for (int i = 0; i < world_size; i++) {
      int element_proc = min_proc;
      if (i < minus_proc) {
        element_proc += 1;
      }

      if (i == 0) {
        std::copy(vec.begin(), vec.begin() + element_proc, l_vec.begin());
      } else {
        MPI_Send(vec.data() + flug, element_proc, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      flug += element_proc;
    }
  } else {
    MPI_Status status;
    MPI_Recv(l_vec.data(), l_n, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  int l_max = 0;
  if (l_n > 1) {
    for (int i = 0; i < l_n - 1; i++) {
      int diff = std::abs(l_vec[i + 1] - l_vec[i]);
      if (diff > l_max) {
        l_max = diff;
      }
    }
  }

  if (world_size > 1) {
    if (l_n > 0) {
      int send_val = l_vec[l_n - 1];
      int recv_val;
      int send_to = (world_rank < world_size - 1) ? world_rank + 1 : MPI_PROC_NULL;
      int recv_from = (world_rank > 0) ? world_rank - 1 : MPI_PROC_NULL;

      MPI_Sendrecv(&send_val, 1, MPI_INT, send_to, 1, &recv_val, 1, MPI_INT, recv_from, 1, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);

      if (world_rank > 0 && recv_from != MPI_PROC_NULL) {
        int boun_diff = std::abs(l_vec[0] - recv_val);
        if (boun_diff > l_max) {
          l_max = boun_diff;
        }
      }
    } else {
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  if (world_rank == 0) {
    int g_max = l_max;
    for (int i = 1; i < world_size; i++) {
      int get_element;
      MPI_Status status;
      MPI_Recv(&get_element, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
      if (get_element > g_max) {
        g_max = get_element;
      }
    }
    GetOutput() = g_max;
  } else {
    MPI_Send(&l_max, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
