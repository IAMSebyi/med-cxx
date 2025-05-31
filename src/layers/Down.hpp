#pragma once

#include "BaseLayer.hpp"
#include "DoubleConv.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// Down-sampling block: MaxPool2d then DoubleConv
class DownImpl : public BaseLayer {
public:
    // Constructor
    DownImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward(torch::Tensor x) override;
    
private:
    // Layers
    torch::nn::MaxPool2d pool{nullptr};
    DoubleConv conv{nullptr};
};
TORCH_MODULE(Down);

} // namespace layers
} // namespace med
