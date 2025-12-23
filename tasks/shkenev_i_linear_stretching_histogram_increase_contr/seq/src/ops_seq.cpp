#include "shkenev_i_linear_stretching_histogram_increase_contr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

ShkenevIlinerStretchingHistIncreaseContrSEQ::ShkenevIlinerStretchingHistIncreaseContrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }

  return std::all_of(GetInput().begin(), GetInput().end(), [](int val) { return val >= 0 && val <= 255; });
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  if (input.empty()) {
    output.clear();
    return true;
  }

  auto min_it = std::ranges::min_element(input);
  auto max_it = std::ranges::max_element(input);
  int min_val = *min_it;
  int max_val = *max_it;

  if (max_val > min_val) {
    int range = max_val - min_val;
    for (size_t i = 0; i < input.size(); ++i) {
      output[i] = (input[i] - min_val) * 255 / range;
    }
  } else {
    output = input;
  }

  return true;
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_linear_stretching_histogram_increase_contr
