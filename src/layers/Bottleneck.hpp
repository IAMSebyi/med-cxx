#pragma once

#include <torch/torch.h>

namespace med {

namespace layers {

// Bottleneck block (for ResNet-50/101/152)
class Bottleneck : public torch::nn::Module {
public:
    static const int expansion = 4;

    // Constructor
    Bottleneck(int inPlanes, int planes, int stride = 1, torch::nn::Sequential downsample_ = {});
    
    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const Bottleneck& layer) {
        os << "Bottleneck block (for ResNet-50/101/152)";
        return os;
    }
private:
    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Sequential downsample;
};

} // namespace layers

} // namespace med
