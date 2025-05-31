#pragma once

#include "BaseLayer.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// Basic residual block (for ResNet-18/34)
class BasicBlock : public BaseLayer {
public:
    static const int expansion = 1;

    // Constructor
    BasicBlock(int inPlanes, int planes, int stride = 1, torch::nn::Sequential downsample_={});

    // Forward pass
    torch::Tensor forward(torch::Tensor x) override;
    
private:
    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample;
};

} // namespace layers
} // namespace med