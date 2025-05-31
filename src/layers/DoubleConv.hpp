#pragma once

#include "BaseLayer.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// Double convolution: two 3x3 conv layers with ReLU
class DoubleConvImpl : public BaseLayer {
public:
    // Constructor
    DoubleConvImpl(int inChannels, int outChannels);

    // Forward pass
    torch::Tensor forward(torch::Tensor x) override;

private:
    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
};
TORCH_MODULE(DoubleConv);

} // namespace layers
} // namespace med
