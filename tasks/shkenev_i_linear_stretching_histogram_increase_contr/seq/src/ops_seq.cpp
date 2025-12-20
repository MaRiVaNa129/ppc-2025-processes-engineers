#include "shkenev_i_linear_stretching_histogram_increase_contr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "shkenev_i_linear_stretching_histogram_increase_contr/common/include/common.hpp"
#include "util/include/util.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

ShkenevIlinerStretchingHistIncreaseContrSEQ::ShkenevIlinerStretchingHistIncreaseContrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  for (int val : GetInput()) {
    if (val < 0 || val > 255) {
      return false;
    }
  }

  return true;
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool ShkenevIlinerStretchingHistIncreaseContrSEQ::RunImpl() {
  auto &input = GetInput();
  auto &output = GetOutput();

  int min_val = *std::min_element(input.begin(), input.end());
  int max_val = *std::max_element(input.begin(), input.end());

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
