#pragma once

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

class ShkenevImatvectUsingVerticalRibbonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShkenevImatvectUsingVerticalRibbonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool HandleSmallMatrixCase(int rank, int rows, int cols);
  void SendDataToProcesses(int world_size, int rows, int cols);
  void ReceiveDataFromProcess0(int rows, int local_cols, std::vector<double> &local_matrix,
                               std::vector<double> &local_vector);
  void GatherAndBroadcastResults(int rank, int rows, const std::vector<double> &local_result,
                                 std::vector<double> &result);
};
}  // namespace shkenev_i_matvect_using_vertical_ribbon
