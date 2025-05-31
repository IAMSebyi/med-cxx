#pragma once

#include "BaseLayer.hpp"
#include "DoubleConv.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// Up-sampling block: ConvTranspose2d then DoubleConv with skip connection
class UpImpl : public BaseLayer {
public:
    // Constructor
    UpImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward_(torch::Tensor x1, torch::Tensor x2);
    torch::Tensor forward(torch::Tensor x) override {
        throw std::runtime_error("UpImpl::forward() should not be called directly, use forward_(x1, x2) instead.");
    }
    
private:
    // Layers
    torch::nn::ConvTranspose2d up{nullptr};
    DoubleConv conv{nullptr};
};
TORCH_MODULE(Up);

} // namespace layers
} // namespace med
