#pragma once

#include "common/ArgParser.hpp"
#include "common/Exception.hpp"
#include "common/Utils.hpp"
#include "common/Visualizer.hpp"
#include "evaluation/Benchmark.hpp"
#include "data/ImageLoader.hpp"
#include "models/BaseModel.hpp"
#include <memory>
#include <string>
#include <torch/torch.h>

namespace med {
namespace trainer {

// Abstract base trainer
class BaseTrainer {
public:
    // Constructor: pass a shared_ptr to a BaseModel, plus the parsed config
    BaseTrainer(std::shared_ptr<models::BaseModel> model,
                const common::Config& cfg);

    virtual ~BaseTrainer() = default;

    // Train the model (pure virtual)
    virtual void train() = 0;

    // Evaluate (inference + metrics) (pure virtual)
    virtual void evaluate() = 0;

protected:
    std::shared_ptr<models::BaseModel> model;
    const common::Config& cfg;
    torch::Device device;

    // Utility: create (and return) a torch::optim::Adam for the given model
    torch::optim::Adam makeOptimizer();

    // Utility: print a progress bar (uses common::printProgressBar)
    void printProgress(size_t current, size_t total);
};

} // namespace trainer
} // namespace med
