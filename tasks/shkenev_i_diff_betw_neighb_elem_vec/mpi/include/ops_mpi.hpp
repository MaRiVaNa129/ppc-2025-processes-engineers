#pragma once

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

class ShkenevIDiffBetwNeighbElemVecMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShkenevIDiffBetwNeighbElemVecMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int HandleSmallVector(const std::vector<int> &vec, int n);
  void ComputeCountsAndDispls(int n, int world_size, std::vector<int> &cnt, std::vector<int> &disp);
  void ScatterData(const std::vector<int> &vec, const std::vector<int> &cnt, const std::vector<int> &disp,
                   std::vector<int> &l_vec, int world_rank);
  int LocalCompute(const std::vector<int> &l_vec);
  int BoundaryExchange(const std::vector<int> &l_vec, int world_rank, int world_size);
};

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
