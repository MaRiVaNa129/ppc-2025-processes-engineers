#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "shkenev_i_diff_betw_neighb_elem_vec/common/include/common.hpp"
#include "shkenev_i_diff_betw_neighb_elem_vec/mpi/include/ops_mpi.hpp"
#include "shkenev_i_diff_betw_neighb_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shkenev_i_diff_betw_neighb_elem_vec {

class ShkenevIDiffBetwNeighbElemVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int kVectorSize = 1000000;
    input_data_.resize(kVectorSize);

    std::random_device random;
    std::mt19937 gen(random());
    std::uniform_int_distribution<int> dist(0, 1000);

    for (int i = 0; i < kVectorSize; i++) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(ShkenevIDiffBetwNeighbElemVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShkenevIDiffBetwNeighbElemVecMPI, ShkenevIDiffBetwNeighbElemVecSEQ>(
        PPC_SETTINGS_shkenev_i_diff_betw_neighb_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShkenevIDiffBetwNeighbElemVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, ShkenevIDiffBetwNeighbElemVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace shkenev_i_diff_betw_neighb_elem_vec
