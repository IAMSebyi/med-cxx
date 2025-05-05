#pragma once

#include <torch/torch.h>

namespace med {

namespace layers {

// Basic residual block (for ResNet-18/34)
class BasicBlock : public torch::nn::Module {
public:
    static const int expansion = 1;

    // Constructor
    BasicBlock(int inPlanes, int planes, int stride = 1, torch::nn::Sequential downsample_={});

    // Forward pass
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded operator<< for printing layer info
    friend std::ostream& operator<<(std::ostream& os, const BasicBlock& layer) {
        os << "Basic residual block (for ResNet-18/34)";
        return os;
    }
private:
    // Layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample;
};

} // namespace layers

} // namespace med