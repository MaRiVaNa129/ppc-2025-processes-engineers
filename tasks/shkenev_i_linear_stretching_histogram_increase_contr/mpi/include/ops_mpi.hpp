#pragma once

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

class ShkenevIlinerStretchingHistIncreaseContrMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShkenevIlinerStretchingHistIncreaseContrMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shkenev_i_linear_stretching_histogram_increase_contr
