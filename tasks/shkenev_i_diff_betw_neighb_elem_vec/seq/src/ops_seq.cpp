#include "shkenev_i_diff_betw_neighb_elem_vec/seq/include/ops_seq.hpp"

#include <numeric>
#include <random>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

ShkenevIDiffBetwNeighbElemVecSEQ::ShkenevIDiffBetwNeighbElemVecSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::ValidationImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::PreProcessingImpl() {
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::RunImpl() {
  std::vector<int> &vec = GetInput();
  if (vec.size() < 2) {
    GetOutput() = 0;
    return true;
  }

  int max_diff = 0;
  int n = vec.size();
  for (int i = 0; i < n - 1; i++) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }

  GetOutput() = max_diff;
  return true;
}

bool ShkenevIDiffBetwNeighbElemVecSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
