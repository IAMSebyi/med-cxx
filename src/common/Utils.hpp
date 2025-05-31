#pragma once

#include <iostream>
#include <iomanip>

namespace med {
namespace util {

// Print progress bar
void printProgressBar(std::size_t current, std::size_t total, std::size_t barWidth = 50);

} // namespace util
} // namespace med