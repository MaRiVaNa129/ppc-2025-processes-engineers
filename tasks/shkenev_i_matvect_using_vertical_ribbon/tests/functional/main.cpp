#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"
#include "shkenev_i_matvect_using_vertical_ribbon/mpi/include/ops_mpi.hpp"
#include "shkenev_i_matvect_using_vertical_ribbon/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

class ShkenevImatvectUsingVerticalRibbonFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int test_id = std::get<0>(test_param);
    const auto &a = std::get<1>(test_param);
    const auto &b = std::get<2>(test_param);

    return "test_" + std::to_string(test_id) + "_" + std::to_string(a.size()) + "x" +
           (a.empty() ? "0" : std::to_string(a[0].size())) + "_" + std::to_string(b.size()) + "x" +
           (b.empty() ? "0" : std::to_string(b[0].size()));
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const auto &a = std::get<1>(params);
    const auto &b = std::get<2>(params);
    input_data_ = std::make_pair(a, b);

    expected_ = std::get<3>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != expected_[i].size()) {
        return false;
      }

      for (size_t j = 0; j < output_data[i].size(); ++j) {
        if (std::abs(output_data[i][j] - expected_[i][j]) > 1e-9) {
          return false;
        }
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

TestType CreateMatrixTest(int test_id, int rows_a, int cols_a, int cols_b) {
  std::vector<std::vector<double>> a(rows_a, std::vector<double>(cols_a));
  std::vector<std::vector<double>> b(cols_a, std::vector<double>(cols_b));
  std::vector<std::vector<double>> c(rows_a, std::vector<double>(cols_b, 0.0));

  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_a; ++j) {
      a[i][j] = (i * cols_a) + j + 1;
    }
  }

  for (int j = 0; j < cols_a; ++j) {
    for (int k = 0; k < cols_b; ++k) {
      b[j][k] = (j * cols_b) + k + 1;
    }
  }

  for (int i = 0; i < rows_a; ++i) {
    for (int k = 0; k < cols_b; ++k) {
      double sum = 0.0;
      for (int j = 0; j < cols_a; ++j) {
        sum += a[i][j] * b[j][k];
      }
      c[i][k] = sum;
    }
  }

  return std::make_tuple(test_id, a, b, c);
}

const std::array<TestType, 8> kTestParam = {
    CreateMatrixTest(1, 2, 2, 2),

    CreateMatrixTest(2, 3, 3, 1),

    CreateMatrixTest(3, 2, 3, 4),

    CreateMatrixTest(4, 4, 3, 2),

    std::make_tuple(5, std::vector<std::vector<double>>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}}),

    std::make_tuple(6, std::vector<std::vector<double>>{{0, 0, 0}, {0, 0, 0}},
                    std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}},
                    std::vector<std::vector<double>>{{0, 0}, {0, 0}}),

    std::make_tuple(7, std::vector<std::vector<double>>{{2.5}}, std::vector<std::vector<double>>{{3.0}},
                    std::vector<std::vector<double>>{{7.5}}),

    CreateMatrixTest(8, 1, 5, 1)};

TEST_P(ShkenevImatvectUsingVerticalRibbonFuncTests, MatrixMultiplication) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShkenevImatvectUsingVerticalRibbonMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shkenev_i_matvect_using_vertical_ribbon),
                                           ppc::util::AddFuncTask<ShkenevImatvectUsingVerticalRibbonSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shkenev_i_matvect_using_vertical_ribbon));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    ShkenevImatvectUsingVerticalRibbonFuncTests::PrintFuncTestName<ShkenevImatvectUsingVerticalRibbonFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, ShkenevImatvectUsingVerticalRibbonFuncTests, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace shkenev_i_matvect_using_vertical_ribbon
