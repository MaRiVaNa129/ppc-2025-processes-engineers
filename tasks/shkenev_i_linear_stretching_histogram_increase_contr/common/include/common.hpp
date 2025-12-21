#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace shkenev_i_linear_stretching_histogram_increase_contr {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkenev_i_linear_stretching_histogram_increase_contr
