#include "Utils.hpp"

void med::util::printProgressBar(std::size_t current, std::size_t total, std::size_t barWidth) {
    if (total == 0) {
        return;
    }

    // Compute ratio and how many “blocks” to fill
    double ratio = static_cast<double>(current) / static_cast<double>(total);
    std::size_t filled = static_cast<std::size_t>(ratio * barWidth);

    std::cout << '\r' << '[';
    for (std::size_t i = 0; i < barWidth; ++i) {
        if (i < filled) {
            std::cout << '#';
        } else if (i == filled) {
            std::cout << '>';
        } else {
            std::cout << ' ';
        }                  
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100) << "% " << std::flush;

    if (current >= total) {
        std::cout << std::endl;
    }
}
