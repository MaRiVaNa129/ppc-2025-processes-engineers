#include "shkenev_i_matvect_using_vertical_ribbon/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "shkenev_i_matvect_using_vertical_ribbon/common/include/common.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

ShkenevImatvectUsingVerticalRibbonSEQ::ShkenevImatvectUsingVerticalRibbonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType temp_input = in;
  GetInput().swap(temp_input);
  GetOutput() = OutType{};
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::ValidationImpl() {
  const auto &input = GetInput();
  const auto &matrix_a = input.first;
  const auto &vector_b = input.second;

  if (matrix_a.empty()) {
    return false;
  }

  std::size_t rows_a = matrix_a.size();
  std::size_t cols_a = matrix_a[0].size();

  for (std::size_t i = 0; i < rows_a; i++) {
    if (matrix_a[i].size() != cols_a) {
      return false;
    }
  }

  if (vector_b.empty()) {
    return false;
  }

  return vector_b.size() == cols_a;
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &vector_b = GetInput().second;

  std::size_t rows_a = matrix_a.size();
  std::size_t cols_a = matrix_a[0].size();

  std::vector<double> result_vector(rows_a, 0.0);

  for (std::size_t i = 0; i < rows_a; i++) {
    for (std::size_t j = 0; j < cols_a; j++) {
      result_vector[i] += matrix_a[i][j] * vector_b[j];
    }
  }

  GetOutput() = result_vector;

  return true;
}

bool ShkenevImatvectUsingVerticalRibbonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_matvect_using_vertical_ribbon
