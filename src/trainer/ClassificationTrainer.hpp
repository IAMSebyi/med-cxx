#pragma once

#include "BaseTrainer.hpp"
#include <torch/torch.h>
#include <filesystem>
#include <iostream>

namespace med {
namespace trainer {

class ClassificationTrainer : public BaseTrainer {
public:
    ClassificationTrainer(std::shared_ptr<models::BaseModel> model,
                          const common::Config& cfg);

    // Train on a directory of class‐subdirectories
    void train() override;

    // Evaluate on a directory of class‐subdirectories
    void evaluate() override;

private:
    // Build a (filename, label) list from a root directory
    std::vector<std::pair<std::string,int>> makeFileLabelList(const std::string& rootDir);

    // Map class‐name -> label id (0..N‐1)
    std::vector<std::string> classes;
    std::unordered_map<std::string,int> classToIdx;
};

} // namespace trainer
} // namespace med