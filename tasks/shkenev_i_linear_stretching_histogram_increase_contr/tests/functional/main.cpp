#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"
#include "shkenev_i_linear_stretching_histogram_increase_contr/mpi/include/ops_mpi.hpp"
#include "shkenev_i_linear_stretching_histogram_increase_contr/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

class ShkenevIlinerStretchingHistIncreaseContrFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int test_id = std::get<0>(test_param);
    const auto &input_data = std::get<1>(test_param);
    return "test_" + std::to_string(test_id) + "_size_" + std::to_string(input_data.size());
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<1>(params);
    expected_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty() && !expected_.empty()) {
      return true;
    }

    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_[i]) > 1) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TestType CreateStretchingTest(int test_id, int size, int min_v, int max_v, int mid_v) {
  std::vector<int> in(size, mid_v);
  in[0] = min_v;
  if (size > 1) {
    in[size - 1] = max_v;
  }

  std::vector<int> out(size);
  int range = max_v - min_v;

  if (range > 0) {
    for (int i = 0; i < size; ++i) {
      out[i] = (in[i] - min_v) * 255 / range;
    }
  } else {
    out = in;
  }

  return std::make_tuple(test_id, in, out);
}

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(1, std::vector<int>{100, 150, 200}, std::vector<int>{0, 127, 255}),
    CreateStretchingTest(2, 1000, 50, 200, 100),

    std::make_tuple(3, std::vector<int>{0, 255}, std::vector<int>{0, 255}),
    std::make_tuple(4, std::vector<int>{0, 0, 255, 255}, std::vector<int>{0, 0, 255, 255}),

    std::make_tuple(5, std::vector<int>(5, 128), std::vector<int>(5, 128)),
    std::make_tuple(6, std::vector<int>(1, 50), std::vector<int>(1, 50)),

    std::make_tuple(7, std::vector<int>{127, 128}, std::vector<int>{0, 255}),

    std::make_tuple(8, std::vector<int>{10, 20, 30}, std::vector<int>{0, 127, 255}),

    std::make_tuple(9, std::vector<int>{1, 2, 254}, std::vector<int>{0, 1, 255}),

    CreateStretchingTest(10, 5000, 10, 240, 120),

    std::make_tuple(11, std::vector<int>{}, std::vector<int>{})};

TEST_P(ShkenevIlinerStretchingHistIncreaseContrFuncTests, LinearStretching) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ShkenevIlinerStretchingHistIncreaseContrMPI, InType>(
                       kTestParam, PPC_SETTINGS_shkenev_i_linear_stretching_histogram_increase_contr),
                   ppc::util::AddFuncTask<ShkenevIlinerStretchingHistIncreaseContrSEQ, InType>(
                       kTestParam, PPC_SETTINGS_shkenev_i_linear_stretching_histogram_increase_contr));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkenevIlinerStretchingHistIncreaseContrFuncTests::PrintFuncTestName<
    ShkenevIlinerStretchingHistIncreaseContrFuncTests>;

INSTANTIATE_TEST_SUITE_P(LinearStretchingTests, ShkenevIlinerStretchingHistIncreaseContrFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace shkenev_i_linear_stretching_histogram_increase_contr
