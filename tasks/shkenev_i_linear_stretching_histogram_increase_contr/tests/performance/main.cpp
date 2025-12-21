#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"
#include "shkenev_i_linear_stretching_histogram_increase_contr/mpi/include/ops_mpi.hpp"
#include "shkenev_i_linear_stretching_histogram_increase_contr/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

class ShkenevIlinerStretchingHistIncreaseContrPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 10000000;

 protected:
  void SetUp() override {
    input_data_.resize(kSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(50, 200);
    for (size_t i = 0; i < kSize; ++i) {
      input_data_[i] = dist(gen);
    }

    input_data_[0] = 0;
    input_data_[kSize - 1] = 255;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }

    if (output_data.size() != kSize) {
      return false;
    }

    return std::ranges::all_of(output_data, [](int val) { return val >= 0 && val <= 255; });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  std::vector<int> input_data_;
};

TEST_P(ShkenevIlinerStretchingHistIncreaseContrPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ShkenevIlinerStretchingHistIncreaseContrMPI,
                                                       ShkenevIlinerStretchingHistIncreaseContrSEQ>(
    PPC_SETTINGS_shkenev_i_linear_stretching_histogram_increase_contr);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = ShkenevIlinerStretchingHistIncreaseContrPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, ShkenevIlinerStretchingHistIncreaseContrPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkenev_i_linear_stretching_histogram_increase_contr
