#include "BaseTrainer.hpp"

namespace med {
namespace trainer {

BaseTrainer::BaseTrainer(std::shared_ptr<models::BaseModel> model, const common::Config& cfg_) 
: model(std::move(model)), cfg(cfg_),
  device((cfg_.useCUDA && torch::cuda::is_available()) ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU)) 
{
    if (cfg_.useCUDA && !torch::cuda::is_available()) {
        std::cout << "[INFO] CUDA requested but not available. Falling back to CPU.\n";
    }
}


torch::optim::Adam BaseTrainer::makeOptimizer() {
    return torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(cfg.learningRate));
}

void BaseTrainer::printProgress(size_t current, size_t total) {
    med::util::printProgressBar(current, total, cfg.printBarWidth);
}

} // namespace trainer
} // namespace med
