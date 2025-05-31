#pragma once

#include "BaseLayer.hpp"
#include <torch/torch.h>

namespace med {
namespace layers {

// Bottleneck block (for ResNet-50/101/152)
class Bottleneck : public BaseLayer {
public:
    static const int expansion = 4;

    // Constructor
    Bottleneck(int inPlanes, int planes, int stride = 1, torch::nn::Sequential downsample_ = {});
    
    // Forward pass
    torch::Tensor forward(torch::Tensor x) override;
    
private:
    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Sequential downsample;
};

} // namespace layers
} // namespace med
