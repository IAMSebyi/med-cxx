#pragma once

#include "BaseLayer.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// Final 1x1 convolution to map to output channels
class OutConvImpl : public BaseLayer {
public:
    // Constructor
    OutConvImpl(int inChannels, int outChannels);
    
    // Forward pass
    torch::Tensor forward(torch::Tensor x) override;

private:
    // Layers
    torch::nn::Conv2d conv{nullptr};
};
TORCH_MODULE(OutConv);

} // namespace layers
} // namespace med
