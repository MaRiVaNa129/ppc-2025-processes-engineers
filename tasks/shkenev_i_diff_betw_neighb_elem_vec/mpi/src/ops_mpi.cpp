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
    std::copy(vec.begin(), vec.begin() + l_n, l_vec.begin());
    flug += l_n;

    for (int i = 1; i < world_size; i++) {
      int proc_size = min_proc;
      if (i < minus_proc) {
        proc_size += 1;
      }
      MPI_Send(vec.data() + flug, proc_size, MPI_INT, i, 0, MPI_COMM_WORLD);
      flug += proc_size;
    }
  } else {
    MPI_Recv(l_vec.data(), l_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  int l_max = 0;
  for (int i = 0; i < l_n - 1; i++) {
    int diff = std::abs(l_vec[i + 1] - l_vec[i]);
    if (diff > l_max) {
      l_max = diff;
    }
  }

  if (world_size > 1) {
    int send_val;
    if (l_n > 0) {
      send_val = l_vec[l_n - 1];
    } else {
      send_val = 0;
    }
    int recv_val;

    int send_to;
    if (world_rank < world_size - 1) {
      send_to = world_rank + 1;
    } else {
      send_to = MPI_PROC_NULL;
    }

    int recv_from;
    if (world_rank > 0) {
      recv_from = world_rank - 1;
    } else {
      recv_from = MPI_PROC_NULL;
    }

    MPI_Sendrecv(&send_val, 1, MPI_INT, send_to, 1, &recv_val, 1, MPI_INT, recv_from, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    if (world_rank > 0 && l_n > 0) {
      int boun_diff = std::abs(recv_val - l_vec[0]);
      if (boun_diff > l_max) {
        l_max = boun_diff;
      }
    }
  }

  int glob_max;
  MPI_Reduce(&l_max, &glob_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = glob_max;
  }

  return true;
}

bool ShkenevIDiffBetwNeighbElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
