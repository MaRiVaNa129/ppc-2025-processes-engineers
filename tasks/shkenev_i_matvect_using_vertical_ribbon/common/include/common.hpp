#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace shkenev_i_matvect_using_vertical_ribbon {

using InType = std::pair<std::vector<std::vector<double>>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::vector<std::vector<double>>, std::vector<double>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shkenev_i_matvect_using_vertical_ribbon
