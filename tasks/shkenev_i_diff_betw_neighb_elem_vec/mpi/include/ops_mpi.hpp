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
  void SendToAllProcesses(const std::vector<int> &vec, int my_size, int world_size, int base_size, int rem);
  void ReceiveFromRoot(std::vector<int> &local_vec, int my_size);
  void DistributeData(const std::vector<int> &vec, std::vector<int> &local_vec, int world_rank, int my_size,
                      int world_size, int base_size, int rem);
};

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
