#pragma once

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

class ShkenevImatvectUsingVerticalRibbonSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShkenevImatvectUsingVerticalRibbonSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace shkenev_i_matvect_using_vertical_ribbon
