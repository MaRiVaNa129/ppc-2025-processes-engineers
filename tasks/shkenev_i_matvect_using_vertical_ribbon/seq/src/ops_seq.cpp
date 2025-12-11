#include "shkenev_i_matvect_using_vertical_ribbon/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

ShkenevImatvectUsingVerticalRibbonSEQ::ShkenevImatvectUsingVerticalRibbonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &matrix_a = input.first;
  const auto &matrix_b = input.second;

  if (matrix_a.empty()) {
    return false;
  }

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();

  for (size_t i = 0; i < rows_a; i++) {
    if (matrix_a[i].size() != cols_a) {
      return false;
    }
  }

  if (matrix_b.empty()) {
    return false;
  }

  size_t rows_b = matrix_b.size();
  size_t cols_b = matrix_b[0].size();

  for (size_t i = 0; i < rows_b; i++) {
    if (matrix_b[i].size() != cols_b) {
      return false;
    }
  }

  if (rows_b != cols_a) {
    return false;
  }

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  size_t rows_a = matrix_a.size();
  size_t cols_a = matrix_a[0].size();
  size_t cols_b = matrix_b[0].size();

  std::vector<std::vector<double>> result_matrix(rows_a, std::vector<double>(cols_b, 0.0));

  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < cols_a; j++) {
      for (size_t k = 0; k < cols_b; k++) {
        result_matrix[i][k] += matrix_a[i][j] * matrix_b[j][k];
      }
    }
  }

  GetOutput() = result_matrix;

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
