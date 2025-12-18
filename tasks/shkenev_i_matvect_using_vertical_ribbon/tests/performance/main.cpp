#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"
#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"
#include "shkenev_i_matvect_using_vertical_ribbon/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

class ShkenevImatvectUsingVerticalRibbonPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 9000;

 protected:
  void SetUp() override {
    matrix_a_ = std::vector<std::vector<double>>(kSize, std::vector<double>(kSize));
    vector_b_ = std::vector<double>(kSize);

    for (size_t i = 0; i < kSize; ++i) {
      for (size_t j = 0; j < kSize; ++j) {
        if ((i + j) % 5 == 0) {
          matrix_a_[i][j] = static_cast<double>((i * kSize) + j) * 0.001;
        } else {
          matrix_a_[i][j] = 0.0;
        }
      }
      vector_b_[i] = static_cast<double>(i) * 0.002;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == kSize;
  }

  InType GetTestInputData() final {
    return std::make_pair(matrix_a_, vector_b_);
  }

 private:
  std::vector<std::vector<double>> matrix_a_;
  std::vector<double> vector_b_;
};

TEST_P(ShkenevImatvectUsingVerticalRibbonPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShkenevImatvectUsingVerticalRibbonMPI, ShkenevImatvectUsingVerticalRibbonSEQ>(
        PPC_SETTINGS_shkenev_i_matvect_using_vertical_ribbon);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = ShkenevImatvectUsingVerticalRibbonPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, ShkenevImatvectUsingVerticalRibbonPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkenev_i_matvect_using_vertical_ribbon
